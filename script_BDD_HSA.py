
# Chargement des librairies 
import re
import yaml
import os
import oracledb
import pandas as pd
import numpy as np

# Chargement des modules (algorithme de TAL développé par Adrien)
from gavroche.inference_NER import algo_adrien
from nettoyage_textes.script_nettoyage import preprocessing,apply_type_doc


# Chargement des librairies Medkit : https://github.com/TeamHeKA/medkit/tree/main/medkit/text/context
from pathlib import Path
from medkit.core.text import TextDocument
from medkit.core import Pipeline, PipelineStep
from medkit.text.preprocessing import RegexpReplacer
from medkit.text.segmentation import SentenceTokenizer, SyntagmaTokenizer
from medkit.text.ner import RegexpMatcher, RegexpMatcherRule
from medkit.text.context import NegationDetector, NegationDetectorRule,FamilyDetector, FamilyDetectorRule



def extraire_patients(con_oracle) :
    
    """Extraction du profil du patient : identifiant, sexe et age (date de naissance)  """
    
    cursor = con_oracle.cursor()
    
    req_patients="""select ID_PAT, sexe, to_char(datenais,'YYYY') as date_de_naissance from  et_10322451.ehop_patient"""
    
    cursor.execute(req_patients)
    
    patients = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    patients['DATE_DE_NAISSANCE'] = pd.to_datetime(patients['DATE_DE_NAISSANCE'])
    return patients

def extraire_sejours_par_date_entree(con_oracle) :
    
    """Extraction des séjours du datamart trié par date d'entrée"""
    
    cursor = con_oracle.cursor()
    
    req_sejours ="""select * from et_10322451.ehop_sejour sej
    order by id_pat,date_entree,date_sortie"""
    
    cursor.execute(req_sejours)
    
    sejours = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    return sejours

def est_consecutif(Y,X):
    
    """Etablis si des séjours (X et Y) sont proches temporellement"""
    
    return (pd.Timedelta(0) <= Y.DATE_ENTREE - X.DATE_SORTIE <= pd.Timedelta(days=1) 
            or X.DATE_ENTREE < Y.DATE_ENTREE < X.DATE_SORTIE 
            or Y.DATE_SORTIE == X.DATE_SORTIE 
            or X.DATE_ENTREE == Y.DATE_ENTREE)

def recherche_sejours_consecutifs(con_oracle):
    
    """Création d'une table de séjours proche temporellement"""
    
    sejours = extraire_sejours_par_date_entree(con_oracle)
    
    # Création d'un nouveau dataframe : ID_SEJ -> ancien identifiant du séjour, ID_SEJ_PRINC -> nouvel identifiant de séjour
    sejours_consecutifs = pd.DataFrame(columns=['ID_PAT','ID_SEJ','ID_SEJ_PRINC'])

    # séjour principal : le nouvel identifiant du séjour qui sera identiques pour tous les séjours consécutifs d'un même patient
    # X, Y : les séjours que l'on comparent 
    # X est le séjour précédent le plus proche temporellement de Y
    sejour_principal = X = sejours.loc[0]

    for i in range (len(sejours)-1) :

        # Séjour suivant
        Y = sejours.loc[i+1]


        # Si concomitant : enregistrement de X et Y dans la table des séjours consécutifs
        if Y['ID_PAT'] == X['ID_PAT'] and est_consecutif(Y,X):
            index = len(sejours_consecutifs)
            sejours_consecutifs.loc[index] = [sejour_principal.ID_PAT,Y.ID_SEJ,sejour_principal.ID_SEJ]
            
            if X.DATE_SORTIE < Y.DATE_SORTIE :
                X = Y

        # Si non concomitants : mise à jour des paramètres
        else :
            sejour_principal = X = Y

            
    return sejours_consecutifs

def recherche_premiers_sejours_HSA(sejours_consecutifs, con_oracle):
    
    """Recherche du premier séjour (et des séjours qui lui sont proches temporellement) 
    pour lequel le patient a eu une HSA (suppression des cas de récidive)"""



    # Enregistrement de la table des séjours consécutifs (sej_cons)
    
    cursor = con_oracle.cursor()

    try:

        req_create_sej_cons="""CREATE TABLE sej_cons(id_pat INT, id_sej INT PRIMARY KEY, id_sej_princ INT)"""
        cursor.execute(req_create_sej_cons) 

        req_insert_sej_cons = """INSERT INTO sej_cons (id_pat,id_sej,id_sej_princ) VALUES (:1, :2, :3)"""
        for index, data in sejours_consecutifs.iterrows():
            cursor.execute(req_insert_sej_cons,(int(data['ID_PAT']),int(data['ID_SEJ']),int(data['ID_SEJ_PRINC'])))
        con_oracle.commit()


    # Gestion des erreurs : si la table exite déja
    except oracledb.DatabaseError as e : 
        
        print("Erreur lors de la création de la table sej_cons ")
        error_obj, = e.args
        list_true_input=["y", "yes", "oui", "o"]

        if error_obj.code == 955: 
            suppr=input('La table sej_cons existe déjà, suppression de la table existante ? y/n ')
            if str.lower(suppr) in list_true_input:
                
                cursor.execute(""" DROP TABLE sej_cons """)
                cursor.execute(req_create_sej_cons) 
                
                req_insert_sej_cons= """INSERT INTO sej_cons (id_pat,id_sej,id_sej_princ) VALUES (:1, :2, :3)"""
                for index, data in sejours_consecutifs.iterrows():
                    cursor.execute(req_insert_sej_cons,(int(data['ID_PAT']),int(data['ID_SEJ']),int(data['ID_SEJ_PRINC'])))
                con_oracle.commit()
                
            else :
                use_last=input("Utiliser la table sej_cons déjà existante ? y/n ")
                if str.lower(use_last) not in list_true_input :
                    print("La table sej_cons existe déjà, et l'utilisateur n'a pas souhaité l'utiliser.")
                    raise

        else : raise

    print("Enregistrement des séjours consécutifs : OK")

    # Enregistrement de la liste des séjours pour une hémoragie sous arachoidienne (HSA) déterminer par Pacome
    
    sejours_HSA = pd.read_excel("PDS_HSA_SEJ_07032025.xlsx")
    cursor = con_oracle.cursor()

    try:

        req_create_sejours_HSA = """CREATE TABLE sej_hsa (id_pat INT,id_sej INT PRIMARY KEY)"""
        cursor.execute(req_create_sejours_HSA)
        for index, data in sejours_HSA.iterrows():
            cursor.execute("""INSERT INTO sej_hsa (id_pat, id_sej) VALUES (:1, :2)""", (int(data['ID_PAT']), int(data['ID_SEJ'])))
        con_oracle.commit()

    except oracledb.DatabaseError as e : 
        
        print("Erreur lors de la création de la table sej_hsa ")
        error_obj, = e.args
        list_true_input=["y", "yes", "oui", "o"]

        if error_obj.code == 955: 
            suppr=input('La table sej_hsa existe déjà, suppression de la table existante ? y/n ')
            if str.lower(suppr) in list_true_input:
                
                cursor.execute(""" DROP TABLE sej_hsa """)
                cursor.execute(req_create_sejours_HSA) 
                for index, data in sejours_HSA.iterrows():
                    cursor.execute("""INSERT INTO sej_hsa (id_pat, id_sej) VALUES (:1, :2)""", (int(data['ID_PAT']), int(data['ID_SEJ'])))
                con_oracle.commit()
                
            else :
                use_last=input("Utiliser la table sej_hsa déjà existante ? y/n ")
                if str.lower(use_last) not in list_true_input :
                    print("La table sej_hsa existe déjà, et l'utilisateur n'a pas souhaité l'utiliser.")
                    raise

        else : raise

    print("Enregistrement des séjours concernant uen HSA : OK")

    # Enregistrement d'une table intermédiaire combinants les deux précédentes :
    # HSA = 1 si c'est un séjour déterminer par Pacome pour une HSA , 0 sinon
    # ID_SEJ_PRINC : nouvel identifiant du séjour
    
    cursor = con_oracle.cursor()
    
    try:

        req_create_sej_cons_hsa="""create table sej_cons_hsa as
        select sej.id_pat,case when scons.id_sej_princ is NULL then sej.id_sej else scons.id_sej_princ end as id_sej_princ,sej.id_sej,sej.date_entree,sej.date_sortie,case when shsa.id_sej is NULL then 0 ELSE 1 end as HSA 
        from et_10322451.ehop_sejour sej left join sejours_hsa shsa on  sej.id_sej = shsa.id_sej
        left join sej_cons scons on sej.id_sej = scons.id_sej
        order by id_pat,id_sej_princ,date_entree"""
        cursor.execute(req_create_sej_cons_hsa)
        con_oracle.commit()
        

    except oracledb.DatabaseError as e : 
        
        print("Erreur lors de la création de la table sej_cons_hsa")
        error_obj, = e.args
        list_true_input=["y", "yes", "oui", "o"]

        if error_obj.code == 955: 
            suppr=input('La table sej_cons_hsa existe déjà, suppression de la table existante ? y/n ')
            if str.lower(suppr) in list_true_input:
                
                cursor.execute(""" DROP TABLE sej_cons_hsa """)
                cursor.execute(req_create_sej_cons_hsa)
                con_oracle.commit()
                
            else :
                use_last=input("Utiliser la table sej_cons_hsa déjà existante ? y/n ")
                if str.lower(use_last) not in list_true_input :
                    print("La table sej_cons_hsa existe déjà, et l'utilisateur n'a pas souhaité l'utiliser.")
                    raise

        else : raise

    print("Enregistrement de la table sejours_consécutifs_HSA : OK")

    #Recherche du premier séjour concernant une HSA à l'aide des tables précédentes

    cursor = con_oracle.cursor()
    
    req_prem_sej_hsa ="""
    select es.id_pat, schsa.id_sej_princ, es.id_sej, es.date_entree,es.date_sortie, es.mode_entree,es.mode_sortie, es.uf_entree,es.uf_sortie,es.urgences,es.chemin,es.type_sej
        from et_10322451.ehop_sejour es inner join sej_cons_hsa schsa on es.id_sej = schsa.id_sej
        where id_sej_princ in (select id_sej_princ 
        from (select id_pat, min(date_entree) as date_premier_sej from sej_cons_hsa where hsa = 1 group by id_pat) date_min
        inner join sej_cons_hsa sch on sch.id_pat = date_min.id_pat where date_entree = date_premier_sej)
        order by es.id_pat, schsa.id_sej_princ, es.date_entree"""
    cursor.execute(req_prem_sej_hsa)
    premiers_sejours_HSA = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])

    # Suppression des tables qui ne sont plus utiles
    try :
        cursor.execute("DROP TABLE sej_cons")
        cursor.execute("DROP TABLE sej_hsa")
        cursor.execute("DROP TABLE sej_cons_hsa")
    except Exception as _: 
        pass

    return premiers_sejours_HSA

def creer_liste_services(liste_uf,dictionnaire_uf_services):
    
    """Création de la liste des services parcourus pour un patient à partir de sa liste d'unités fonctionnelles parcourus (list_uf)
    et d'un dictionnaire de correspondance (dictionnaire_uf_services)"""
    
    if pd.isna(liste_uf) or liste_uf == '' :
        return None 
    else :
        liste_services = ''
        service_precedent = None
        for uf in liste_uf.split(';') : 
            uf = str(uf)[:4] # Un cas  : UF de 6 lettres (erreur) 
            service = dictionnaire_uf_services[dictionnaire_uf_services['Source'] == uf]['SERVICE_SYNTH'].values[0]
            if service != service_precedent: 
                liste_services = liste_services + ' - ' + service
            service_precedent = service
        return liste_services
    
    
def exclure_sejours_non_MCO(prem_sej_hsa):
    
    """Suppression des séjours en rééducation ou en psychiatrie : seule la partie d’hospitalisation est conservée 
    (médecine, chirurgie, obstétrique), la psychiatrie et la rééducation étant considérées comme des prises en charge
    post-hospitalisation. """

    # 1. Détermination des services parcourus pour chaque liste d'unités fonctionnelles parcourus
    
    
    # Chargement du dictionnaire des correspondances
    dictionnaire_uf_services = pd.read_excel("UF.xlsx")[["Source","SERVICE_SYNTH"]].drop_duplicates()
    dictionnaire_uf_services['Source'] = dictionnaire_uf_services['Source'].astype('str')

    # Détermination du service d'entrée et de sortie
    prem_sej_hsa = prem_sej_hsa.merge(dictionnaire_uf_services, left_on="UF_ENTREE",right_on="Source",how = "left").drop('Source', axis=1).rename(columns={'SERVICE_SYNTH': 'SERVICE_ENTREE'})
    prem_sej_hsa = prem_sej_hsa.merge(dictionnaire_uf_services, left_on="UF_SORTIE",right_on="Source",how = "left").drop('Source', axis=1).rename(columns={'SERVICE_SYNTH': 'SERVICE_SORTIE'})
    
    # Détermination des unités fonctionelles parcourus lorsque celle ci est vide
    # à l'aide de l'unité fontionnelle d'entrée et de sortie
    
    conditions = [(prem_sej_hsa['CHEMIN'].isna()) & (prem_sej_hsa['UF_ENTREE'].notna()) & (prem_sej_hsa['UF_SORTIE'].notna()),
                  (prem_sej_hsa['CHEMIN'].isna()) & (prem_sej_hsa['UF_ENTREE'].notna())]
    choix = [prem_sej_hsa['UF_ENTREE'] + ";" +  prem_sej_hsa['UF_SORTIE'],prem_sej_hsa['UF_ENTREE']]
    prem_sej_hsa['UF_PARCOURUS'] = np.select(conditions, choix, default = prem_sej_hsa['CHEMIN'])

    # Création de la liste des services parcourus pour chaque patient
    prem_sej_hsa['SERVICES_PARCOURUS'] = prem_sej_hsa['UF_PARCOURUS'].apply(lambda liste_uf : creer_liste_services(liste_uf, dictionnaire_uf_services))
    prem_sej_hsa['SERVICES_PARCOURUS'] = prem_sej_hsa['SERVICES_PARCOURUS'].str.replace(r'^ - ', '', regex=True)
    
    # 2. Suppression des séjours en rééducation ou en psychiatrie
    
    # Identification des séjours concernés
    sejours_non_mco = prem_sej_hsa[prem_sej_hsa['SERVICES_PARCOURUS'].str.split(' - ').isin([['REEDUC'],['PSY']])]
    sejours_non_mco_first_date = sejours_non_mco.groupby('ID_PAT').agg(DATE_ENTREE_NON_MCO=('DATE_ENTREE', 'min'))
    sejours_non_mco = prem_sej_hsa[prem_sej_hsa['ID_PAT'].isin(sejours_non_mco['ID_PAT'])]
    sejours_non_mco = sejours_non_mco.merge(sejours_non_mco_first_date,on='ID_PAT',how ='inner')
     
    # Extractuion de la partie du séjour avant le passage en rééduction / psychiatrie
    sejours_non_mco = sejours_non_mco[(sejours_non_mco['DATE_SORTIE'] <= sejours_non_mco['DATE_ENTREE_NON_MCO'] + pd.Timedelta(hours=1)) | ((sejours_non_mco['DATE_ENTREE'] <= sejours_non_mco['DATE_ENTREE_NON_MCO']) & (pd.isna(sejours_non_mco['DATE_SORTIE'])))]
    sejours_non_mco.drop('DATE_ENTREE_NON_MCO', axis=1, inplace=True)
    
    # Marquage des séjours qui ont un paasage en rééducation / psychiatrie ensuite ( 38 patients concernés)
    sejours_non_mco['SORTIE_REEDUC'] = True
    
    # Création d'une liste final ne contenant que la partie du séjour en hospitalisation
    sejours_mco = prem_sej_hsa[~prem_sej_hsa['ID_PAT'].isin(sejours_non_mco['ID_PAT'])].copy()
    sejours_mco['SORTIE_REEDUC'] = False
    sejours = pd.concat([sejours_mco, sejours_non_mco], axis=0, ignore_index=True)

    # Suppression des séjours qui ont été déteminer comme concomitant par erreur
    # Ces séjour ont été déterminer suite à une analyse des outliers (durée ou nombre de services parcourus supérieur à la normal)
    # ou lorsque lorsque la sortie d'hospitalisation était manquante (variable essentielle à notre analyse)
    
    sejours = sejours[~sejours['ID_SEJ'].isin([8919817, 8919818, 8919819, 8919821,8919823,8919824,8919825,8919826,
                                               12907745,12907747,12907748,
                                               17896996,17993638,17997205,18151741,18298882,18289902,18990679,19328636,
                                               7268672,
                                               16220182,16220183,16220184,16220185,
                                               15071998,
                                               17079358])]
    
    return sejours.sort_values('ID_PAT')
        
def aggreger_sejours(sejours_MCO,patients) :
    
    """ Fusion des séjours concomitants"""
    

    dictionnaire_uf_services = pd.read_excel("UF.xlsx")[["Source","SERVICE_SYNTH"]].drop_duplicates()
    dictionnaire_uf_services['Source'] = dictionnaire_uf_services['Source'].astype('str')
    
    
    # Agrégation des séjours
    # Date d'entrée : date la plus ancienne parmi les séjours à aggréger (date de sortie : la plus récente)
    sej_aggr = sejours_MCO.groupby('ID_PAT').agg(DATE_ENTREE = ('DATE_ENTREE',"min"), DATE_SORTIE=('DATE_SORTIE',"max")).reset_index()
    
    
    # Information concernant l'entrée du séjour récupérée du séjour dont la date d'entrée est la plus ancienne
    # (information sur la sortie  : séjour dont la date de sortie est la plus récente)
    premier_sejour = sejours_MCO.merge(sej_aggr, on=["ID_PAT", "DATE_ENTREE"], how="inner")
    premier_sejour = premier_sejour[['ID_PAT', 'DATE_ENTREE', 'MODE_ENTREE', 'UF_ENTREE', 'SERVICE_ENTREE','URGENCES','TYPE_SEJ']]
    dernier_sejour = sejours_MCO.merge(sej_aggr, on=["ID_PAT", "DATE_SORTIE"], how="inner")
    dernier_sejour = dernier_sejour[['ID_PAT', 'DATE_SORTIE', 'MODE_SORTIE', 'UF_SORTIE', 'SERVICE_SORTIE','SORTIE_REEDUC']]
    sejours = premier_sejour.merge(dernier_sejour, on='ID_PAT',how='inner')
    
    
    # Suprression des séjours en doublons pour un patient
    # (cas particuliers où deux séjours ont la même date d'entrée la plus ancienne / la même date de sortie la plus récente)
    sejours = sejours.drop_duplicates(subset='ID_PAT', keep='first') 

    # Création de nouvelles variables : Durée d'hospitalisation, age lors de l'hospitalisation, services parcourus et leurs nombres
    sejours['DUREE'] = sejours['DATE_SORTIE'] - sejours['DATE_ENTREE']
    sejours['DUREE'] = sejours['DUREE'].dt.total_seconds() / (60 * 60 * 24 )
    sejours = sejours.merge(patients, on='ID_PAT', how='inner')
    sejours['AGE'] = [row['DATE_ENTREE'].year - row['DATE_DE_NAISSANCE'].year for index,row in sejours.iterrows()]
    sejours['UF_PARCOURUS'] = sejours_MCO.sort_values(by = ['ID_PAT','DATE_ENTREE']).groupby('ID_PAT')['UF_PARCOURUS'].apply(lambda x: ';'.join(x)).reset_index(drop=True)
    sejours['SERVICES_PARCOURUS'] = sejours['UF_PARCOURUS'].apply(lambda list_uf : creer_liste_services(list_uf, dictionnaire_uf_services))
    sejours['SERVICES_PARCOURUS'] = sejours['SERVICES_PARCOURUS'].str.replace(r'^ - ', '', regex=True)
    sejours['NB_SERVICES_PARCOURUS'] = sejours['SERVICES_PARCOURUS'].apply(lambda x : len(x.split(' - ')))
    
    
    return sejours[['ID_PAT','AGE','DATE_ENTREE','DATE_SORTIE','DUREE','MODE_ENTREE','MODE_SORTIE','TYPE_SEJ','UF_ENTREE','SERVICE_ENTREE','UF_SORTIE','SERVICE_SORTIE','URGENCES','NB_SERVICES_PARCOURUS','UF_PARCOURUS','SERVICES_PARCOURUS','SORTIE_REEDUC']]

def charger_sejours(con_oracle,sejours_aggreges,sejours_MCO):
    
    cursor = con_oracle.cursor()
     
    try:

        req_create_sej_mco = """CREATE TABLE sej_mco(id_pat INT, id_sej INT PRIMARY KEY)"""
        cursor.execute(req_create_sej_mco)        
        for index, data in sejours_MCO.iterrows():
            cursor.execute("""INSERT INTO sej_mco(id_pat,id_sej) VALUES (:1, :2)""", (int(data['ID_PAT']), int(data['ID_SEJ'])))
        con_oracle.commit()

    except oracledb.DatabaseError as e : 
        
        print("Erreur lors de la création de la table sej_mco")
        error_obj, = e.args
        list_true_input=["y", "yes", "oui", "o"]

        if error_obj.code == 955: 
            suppr=input('La table sej_mco existe déjà, suppression de la table existante ? y/n ')
            if str.lower(suppr) in list_true_input:
                
                cursor.execute(""" DROP TABLE sej_mco """)
                cursor.execute(req_create_sej_mco)               
                for index, data in sejours_MCO.iterrows():
                    cursor.execute( """INSERT INTO sej_mco(id_pat,id_sej) VALUES (:1, :2)""", (int(data['ID_PAT']), int(data['ID_SEJ'])))
                con_oracle.commit()
                
            else :
                use_last=input("Utiliser la table sej_mco déjà existante ? y/n ")
                if str.lower(use_last) not in list_true_input :
                    print("La table sej_mco existe déjà, et l'utilisateur n'a pas souhaité l'utiliser.")
                    raise

        else : raise
    
    try:

        req_create_sej_aggr= """CREATE TABLE sej_aggr(id_pat INT PRIMARY KEY, age int, date_entree date, date_sortie date,
                    mode_entree char(4), mode_sortie char(4), uf_entree int, service_entree varchar(10), uf_sortie int, service_sortie varchar(10),
                    urgences char(5), type_sejour char(1), uf_parcourus varchar(500), services_parcourus varchar(700), nb_services_parcourus int)"""
        
        cursor.execute(req_create_sej_aggr)
        
        req_insert_sej_aggr = """INSERT INTO sej_aggr(id_pat,age,date_entree,date_sortie,mode_entree,mode_sortie,uf_entree,service_entree,
            uf_sortie,service_sortie,urgences,type_sejour, uf_parcourus,services_parcourus,nb_services_parcourus) VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15)"""
        
        for index, data in sejours_aggreges.iterrows():
            cursor.execute(req_insert_sej_aggr, (int(data['ID_PAT']), int(data['AGE']),data['DATE_ENTREE'],data['DATE_SORTIE'],str(data['MODE_ENTREE']),
                str(data['MODE_SORTIE']),int(data['UF_ENTREE']),data['SERVICE_ENTREE'],int(data['UF_SORTIE']),data['SERVICE_SORTIE'],str(data['URGENCES']),
                str(data['TYPE_SEJ']),str(data['UF_PARCOURUS']),str(data['SERVICES_PARCOURUS']),int(data['NB_SERVICES_PARCOURUS'])))
        con_oracle.commit()

    except oracledb.DatabaseError as e : 
        
        print("Erreur lors de la création de la table sej_aggr")
        error_obj, = e.args
        list_true_input=["y", "yes", "oui", "o"]

        if error_obj.code == 955: 
            suppr=input('La table sej_aggr existe déjà, suppression de la table existante ? y/n ')
            if str.lower(suppr) in list_true_input:
                
                cursor.execute(""" DROP TABLE sej_aggr """)
                cursor.execute(req_create_sej_aggr)
                
                req_insert_sej_aggr = """INSERT INTO sej_aggr(id_pat,age,date_entree,date_sortie,mode_entree,mode_sortie,uf_entree,service_entree,
                uf_sortie,service_sortie,urgences,type_sejour,uf_parcourus,services_parcourus,nb_services_parcourus) VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15)"""
                
                for index, data in sejours_aggreges.iterrows():
                    cursor.execute(req_insert_sej_aggr, (int(data['ID_PAT']), int(data['AGE']),data['DATE_ENTREE'],data['DATE_SORTIE'],str(data['MODE_ENTREE']),
                        str(data['MODE_SORTIE']),int(data['UF_ENTREE']),data['SERVICE_ENTREE'],int(data['UF_SORTIE']),data['SERVICE_SORTIE'],str(data['URGENCES']),
                        str(data['TYPE_SEJ']),str(data['UF_PARCOURUS']),str(data['SERVICES_PARCOURUS']),int(data['NB_SERVICES_PARCOURUS'])))
                con_oracle.commit()
                
            else :
                use_last=input("Utiliser la table sej_aggr déjà existante ? y/n ")
                if str.lower(use_last) not in list_true_input :
                    print("La table sej_aggr existe déjà, et l'utilisateur n'a pas souhaité l'utiliser.")
                    raise

        else : raise

def extraire_poids(table_final, con_oracle):
    
    """Extraction du premier poids enregistré durant le séjour par patient """
    
    cursor = con_oracle.cursor()
    req = """select ess1.id_pat, nombre as poids
    from(select ess1.id_pat, min(date_data) as prem_date from  et_10322451.ehop_entrepot_structure ess1 inner join sej_aggr sej on  ess1.id_pat = sej.id_pat
    where ess1.code_thesaurus = 'DPI-CST' and ess1.code='Poids_Autre' and date_data between date_entree and date_sortie
    group by ess1.id_pat) min_date inner join  et_10322451.ehop_entrepot_structure ess1 on ess1.id_pat = min_date.id_pat
    where date_data = prem_date and ess1.code_thesaurus = 'DPI-CST' and ess1.code='Poids_Autre'"""
    cursor.execute(req)
    poids = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    table_final =  table_final.merge(poids, on=["ID_PAT"], how = "left")
    return table_final

def extraire_score_glasgow(table_final, con_oracle):
    
    """ Extraction du score de Glasgow le plus bas par patient """
    
    cursor = con_oracle.cursor()
    req = """select ess1.id_pat,min(nombre) as glasgow
    from  et_10322451.ehop_entrepot_structure ess1 inner join sej_aggr sej on ess1.id_pat = sej.id_pat
    where ess1.code_thesaurus = 'DPI-CST' and ess1.code='Glasgow_Autre' and date_data between date_entree and date_sortie
    group by ess1.id_pat
    order by id_pat"""
    cursor.execute(req)
    score_glasgow = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    table_final =  table_final.merge(score_glasgow, on=["ID_PAT"], how = "left")
    return table_final
    
def extraire_medicaments(codes_atc, nom_molecule , table_final, con_oracle):
    
    """ Extraction de la première date d'administration de médicaments associé à une molécule par patient """
    
    cursor = con_oracle.cursor()
    req = f"""select str.id_pat,min(date_data) as NOM_VAR
    from et_10322451.ehop_entrepot_structure str inner join theriaque.sp_specialite sp on str.code = sp.SP_CIPUCD  and str.code_thesaurus='UCD'
    inner join sej_aggr sej on str.id_pat = sej.id_pat
    WHERE SP_CATC_CODE_FK in {codes_atc} and date_data between sej.date_entree and sej.date_sortie
    group by str.id_pat"""
    cursor.execute(req)   
    medicaments = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    medicaments.rename(columns={'NOM_VAR': nom_molecule}, inplace=True)
    table_final =  table_final.merge(medicaments, on=["ID_PAT"], how = "left")
    return table_final

def extraire_actes(codes_ccam, nom_acte, table_final,con_oracle):
    
    """ Extraction de la première date de réalisation d'un acte par patient """
    
    cursor = con_oracle.cursor()
    req = f"""select sej.id_pat,min(date_data) as NOM_VAR
    from  et_10322451.ehop_entrepot_structure str inner join sej_aggr sej on sej.id_pat = str.id_pat
    where code_thesaurus ='ccam'  and code in {codes_ccam} and date_data between trunc(sej.date_entree) and sej.date_sortie
    group by sej.id_pat"""
    cursor.execute(req)
    actes = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    actes.rename(columns={'NOM_VAR': nom_acte}, inplace=True)
    table_final =  pd.merge(table_final, actes, on=["ID_PAT"], how = "left")
    return table_final


def extraire_resultats_prelevements(code_labo, code_sql, nom_molecule, table_final,con_oracle):
    
    """ Extraction de la première date où le résultat du prélèvement était normal, bas ou élevé par patient,
    date de prélèvement récupéré en priorité"""
    
    cursor = con_oracle.cursor()
    req = f""" select id_pat, min(date_data) as NOM_VAR
    from(select ess1.id_pat, case when ess2.date_data IS NULL then ess1.date_data else ess2.date_data end as date_data,sej.date_entree, sej.date_sortie
    from  et_10322451.ehop_entrepot_structure ess1 left join et_10322451.ehop_entrepot_structure ess2 on ess1.id_entrepot = ess2.id_entrepot and ess2.code='date_prelevement'
    inner join sej_aggr sej on ess1.id_pat = sej.id_pat
    where ess1.code_thesaurus = 'labo' and ess1.code = :1 and ess1.nombre {code_sql})
    where date_data between date_entree and date_sortie
    group by id_pat """
    cursor.execute(req,([code_labo]))
    resultats_prelevements = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    resultats_prelevements.rename(columns={'NOM_VAR': nom_molecule}, inplace=True)
    table_final =  pd.merge(table_final, resultats_prelevements, on=["ID_PAT"], how = "left")
    return table_final

def extraire_constantes(code_constante, code_sql, nom_constante, table_final,con_oracle):
    
    """ Extraction de la première date où la mesure de la constante (température, pression artérielle, etc) était basse, élevé ou normal """
    
    cursor = con_oracle.cursor()
    req = f"""select ess1.id_pat, min(date_data) as nom_var
    from  et_10322451.ehop_entrepot_structure ess1 inner join sej_aggr sej on ess1.id_pat = sej.id_pat
    where ess1.code_thesaurus = 'DPI-CST' and ess1.code = :1 and ess1.nombre {code_sql} and date_data between date_entree and date_sortie
    group by ess1.id_pat"""
    cursor.execute(req,([code_constante]))
    constantes = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    constantes.rename(columns={'NOM_VAR': nom_constante}, inplace=True)
    table_final =  pd.merge(table_final, constantes, on=["ID_PAT"], how = "left")
    return table_final

def extraire_donnees_structurees(sejours_aggreges,con_oracle):
    
    """ Extraction des donnees structurées à l'aide de terminologies médicales (code ATC, CCAM, terminologie locale) """

    table_final = sejours_aggreges[['ID_PAT']]
    
    # Poids / score de Glasgow
    
    table_final = extraire_poids(table_final,con_oracle)
    table_final = extraire_score_glasgow(table_final,con_oracle)
    
    # Medicaments
    
    table_final = extraire_medicaments("('C08CA06')", 'nimodipine', table_final,con_oracle)
    table_final = extraire_medicaments(('N02BE01','N02BE06','N02AJ13','N02BE51','N02AJ01','N02AJ17','N02AJ22','N02BE71'), 'paracetamol', table_final,con_oracle)
    table_final = extraire_medicaments("('C01CA03')", 'noradrenaline', table_final,con_oracle)
    table_final = extraire_medicaments("('C01CE02')", 'milrinone', table_final,con_oracle)
    table_final = extraire_medicaments(('N02AA01','N02AA51','N02AG01'),'morphine', table_final,con_oracle)
    table_final = extraire_medicaments(('N03AX14','N03AX18','N03AE01','N03AA02','N03AG01','N03AB02','N03AB05'), 'antiepideptique', table_final,con_oracle)
    table_final = extraire_medicaments(('N03AX14','N03AX18'), 'antiepideptique_HSA', table_final,con_oracle)
    table_final = extraire_medicaments(('M04AA01','M04AA02','M04AA03','M04AA51'), 'goutte_date', table_final,con_oracle)
    table_final['goutte'] = table_final['goutte_date'].notna().astype('bool')
    table_final.drop('goutte_date', axis=1, inplace =True)

    # Actes 
    
    #ACQH003 : scan +  produit    ACQK001 : scan + sans produit    ACQN001 : IRM  
    codes_ccam = ('ACQH003','ACQK001','ACQN001') 
    table_final = extraire_actes(codes_ccam, 'diagnostic', table_final,con_oracle)

    codes_ccam = "('ABCB001')"
    table_final = extraire_actes(codes_ccam, 'DVE', table_final,con_oracle)

    codes_ccam = "('ABCA002')"
    table_final = extraire_actes(codes_ccam, 'DVP', table_final,con_oracle)

    codes_ccam = ('EAAF004','EAAF903')
    table_final = extraire_actes(codes_ccam, 'angioplastie', table_final,con_oracle)

    codes_ccam = ('GLLD015','GLLD008')
    table_final = extraire_actes(codes_ccam, 'intubation_orotracheale', table_final,con_oracle)

    codes_ccam = ('EABA001','EACA002','EACA003','EACA004','EACA007','EASF007','EASF010','EASF013') 
    table_final = extraire_actes(codes_ccam, 'traitement_AIC', table_final,con_oracle)
    
    # Résultats de prélèvement , messures des constantes

    # Fievre
    table_final = extraire_constantes('Temp_Autre', ' >  37.9 ', 'fievre', table_final,con_oracle)


    # Pression artérielle 
    table_final = extraire_constantes('PASys_Autre', ' between 90 and 140 ', 'PA_normal', table_final,con_oracle)
    table_final = extraire_constantes('PASys_Autre', ' > 140 ', 'PA_eleve', table_final,con_oracle)
    table_final = extraire_constantes('PASys_Autre', ' < 90 ', 'PA_bas', table_final,con_oracle)
    
    # Desaturation en oxygène 
    table_final = extraire_constantes('SpO2_Autre', ' < 95 ', 'desaturation_O2', table_final,con_oracle)
    
    # Sodium
    table_final = extraire_resultats_prelevements( 'NA', ' between 135 and 145 ', 'NA_normal', table_final,con_oracle)
    table_final = extraire_resultats_prelevements( 'NA', ' > 145 ', 'NA_eleve', table_final,con_oracle)
    table_final = extraire_resultats_prelevements( 'NA', ' < 135 ', 'NA_bas', table_final,con_oracle)
    
    # Pression de l'oxygène dnas le sang faible
    table_final = extraire_resultats_prelevements('PO2A', ' < 10 ' , 'PA_O2_bas', table_final,con_oracle)

    # Taux d'hémoglobine bas
    table_final = extraire_resultats_prelevements('HB', ' < 12 ', 'anemie', table_final,con_oracle)
    
    # Glucose 
    table_final = extraire_constantes('GlyCapil_Autre', ' between 2.75 and 6.93 ', 'glucose_normal', table_final,con_oracle)
    table_final = extraire_constantes('GlyCapil_Autre', ' > 6.93 ','glucose_eleve', table_final,con_oracle)   
    table_final = extraire_constantes('GlyCapil_Autre', ' < 2.75 ','glucose_bas', table_final,con_oracle)
    
    return table_final

def extraire_cr_hospi_atcd(con_oracle):
    
    """ Extraction des comptes rendus d'hospitalisation écrits avant la date de sortie du patient 
    (permet d'extraire les antécédents du patient)"""
    
    cursor = con_oracle.cursor()
    req="""select txtid,et.id_pat,et.id_sej,trunc(datesignature) as date_doc, ee.uf, ee.type_doc, theso.code_libelle, titre, et.texte
        from et_10322451.ehop_texte et inner join et_10322451.ehop_entrepot ee on ee.id_entrepot = et.id_entrepot left join edbm_eds.ehop_thesaurus theso 
        on theso.code = ee.type_doc inner join sej_aggr hsa on et.id_pat = hsa.id_pat
        where ((type_doc in ('LN:11490-0','LN:15507-7','LN:34749-2','LN:34112-3','LN:77436-4') 
        and titre not in ('ANES_SSPI','ANES_CRA','Observations d''anesthésie')) or titre = 'Compte rendu CLINICOM')
        and theso.code_thesaurus ='NOMENT_TYPE_DOC' and contexte='texte'
        and (datesignature <= trunc(date_sortie) + 2 or et.id_sej in (select id_sej from sej_mco))"""
    cursor.execute(req)
    cr_hospi = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    return cr_hospi.sort_values(by='ID_PAT').reset_index(drop=True)

def extraire_atcd_avec_transformers(con_oracle):
    
    """ Extraction des mentions de diabète et d'hypertension artérielles dans les comptes rendus (CR) d'hospitalisation 
    à l'aide d'un modèle de transformer """

    cr_hospi = extraire_cr_hospi_atcd(con_oracle)
    
    # Nettoyage des textes

    cr_hospi['TEXTE'] = cr_hospi['TEXTE'].astype(str)
    cr_hospi['TEXTE'] = cr_hospi['TEXTE'].apply(lambda x: re.sub(r"\t|\xa0|\n", "",x))
    cr_hospi['TEXTE'] = cr_hospi['TEXTE'].apply(lambda x: re.sub(r"\s+", " ",x))
    
    # Détermination du type de compte rendus de chaque document  (cr neurologie, de réanimation, etc)
    cr_hospi['type_doc_nlp'] = cr_hospi['TEXTE'].apply(apply_type_doc)
    
    # Suppression des bandeaux (balises, en tete, pied de page) en fonction du type de document déterminé juste au dessus
    cr_hospi['TEXTE_clean'] = cr_hospi['TEXTE'].apply(preprocessing)
    
    
    # Lancement des modèles de transformers permettant l'annotation des documents
    
    algo_adrien(model_path = './gavroche/DrLongformer_bloc2_run0.model', data_annot = cr_hospi)
    algo_adrien(model_path = './gavroche/DrLongformer_bloc3_run0.model', data_annot = cr_hospi)


    table_final = cr_hospi[['ID_PAT']].drop_duplicates().sort_values(by='ID_PAT').reset_index(drop=True)

    # Récupération des annotations de l'hypertension artérielle (HTA) 
    # Seules les mentions à la forme affirmative d'antécédents d'hypertension artérielle sont annotées 'ATCD_HTA'
    resultat = pd.read_json('gavroche/DrLongformer_bloc2_run0.model/resultats.json')
    
    table_final['HTA'] = None
    for idx,data in resultat.iterrows():
        for entity in data['res']:
            if entity['entity_group'] in ['ATCD_HTA'] :
                table_final.at[table_final[table_final['ID_PAT'] == data['id_pat']].index[0],'HTA'] = True

    # Récupération des annotations du diabète  
    # L'annotation des antécédents de diabète prend en compte la détection de négation en annotant 'PAS_DE_DIABETE' si la mention de 
    # diabète est associé à une négation
    resultat = pd.read_json('gavroche/DrLongformer_bloc3_run0.model/resultats.json')
    
    table_final['diabete'] = None
    for idx,data in resultat.iterrows():
        for entity in data['res']:
            if entity['entity_group'] in ['ATCD_DIABETE_TYPE1','ATCD_DIABETE_TYPE2','ATCD_DIABETE_AUTRE'] :
                table_final.at[table_final[table_final['ID_PAT'] == data['id_pat']].index[0],'diabete'] = True
            elif entity['entity_group'] in ['PAS_DE_DIABETE'] :
                table_final.at[table_final[table_final['ID_PAT'] == data['id_pat']].index[0],'diabete'] = False
                
                
def extraire_cr_hospi_evt(con_oracle):
    
     """ Extraction des comptes rendus d'hospitalisation écrits durant le séjour du patient 
     (permet d'extraire les évênements (evt) se produisant durant le séjour) """
      
     cursor = con_oracle.cursor()
     req="""select txtid, et.id_pat,et.id_sej,trunc(datesignature) as date_doc,ee.uf,ee.type_doc, theso.code_libelle, titre, et.texte
        from et_10322451.ehop_texte et inner join  et_10322451.ehop_entrepot ee on ee.id_entrepot = et.id_entrepot left join edbm_eds.ehop_thesaurus theso on theso.code = ee.type_doc
        inner join sej_aggr hsa on et.id_pat = hsa.id_pat
        where ((type_doc in ('LN:11490-0','LN:15507-7','LN:34749-2','LN:34112-3','LN:77436-4') 
        and titre not in ('ANES_SSPI','ANES_CRA','Observations d''anesthésie')) or titre = 'Compte rendu CLINICOM')
        and theso.code_thesaurus ='NOMENT_TYPE_DOC' and contexte='texte'
        and (datesignature between trunc(date_entree) and trunc(date_sortie) + 2  or et.id_sej in (select id_sej from sej_mco))
        order by et.id_pat"""
     cursor.execute(req)
     cr_hospi = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
     return cr_hospi.sort_values('ID_PAT').reset_index(drop=True)
 
def extraire_cr_hospi_neuro(con_oracle) : 
    
    """ Extraction des comptes rendus d'hospitalisation, d'imagerie ou opératoire 
    (permet d'extraire des événements et des caractéristiques de l'anévrisme rompu) """
    
    cursor = con_oracle.cursor()
    req="""select txtid, et.id_pat,ee.uf,ee.uf,ee.type_doc, theso.code_libelle, titre, et.texte
        from et_10322451.ehop_texte et inner join  et_10322451.ehop_entrepot ee on ee.id_entrepot = et.id_entrepot left join edbm_eds.ehop_thesaurus theso on theso.code = ee.type_doc
        inner join sej_aggr hsa on et.id_pat = hsa.id_pat
        where ((type_doc in ('LN:34874-8','LN:11488-4','LN:18748-4','LN:77436-4','LN:11490-0','LN:34112-3','LN:34749-2') or titre = 'Compte rendu CLINICOM')
        and titre not in ('Observations d''anesthésie','ANES_SSPI','ANES_CRA')) and theso.code_thesaurus ='NOMENT_TYPE_DOC' and contexte='texte'
        and (datesignature between trunc(date_entree) and trunc(date_sortie) + 2 or et.id_sej in (select id_sej from sej_mco))
        order by et.id_pat"""
    cursor.execute(req)
    cr_neuro_hospi= pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    return cr_neuro_hospi.sort_values('ID_PAT').reset_index(drop=True)

def extraire_cr_neuro(con_oracle): 
    
    """ Extraction des comptes rendus liés à la neurologie (cr de neuroradiologie ou opératoire) """
    
    cursor = con_oracle.cursor()
    req="""select txtid, et.id_pat,ee.uf,trunc(datesignature) as date_doc,ee.uf,ee.type_doc, theso.code_libelle, titre, et.texte
    from et_10322451.ehop_texte et inner join  et_10322451.ehop_entrepot ee on ee.id_entrepot = et.id_entrepot left join edbm_eds.ehop_thesaurus theso on theso.code = ee.type_doc
    inner join sej_aggr hsa on et.id_pat = hsa.id_pat
    where theso.code_thesaurus ='NOMENT_TYPE_DOC' and type_doc in ('LN:34874-8','LN:11488-4','LN:18748-4','LN:77436-4')
    and (datesignature between trunc(date_entree) and trunc(date_sortie) + 2 or et.id_sej in (select id_sej from sej_mco))"""

    cursor.execute(req)
    cr_neuro = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
    return cr_neuro.sort_values('ID_PAT').reset_index(drop=True)


def pipeline_medkit(cr,regexp_rules):
    
    """ Recherche d'entités dans les comptes rendus (cr) et du contexte associé à celle ci à l'aide d'expression régulière (regex_rules) """
    
    # Transformation des textes en document Medkit
    cr['TEXTE'] = cr['TEXTE'].astype(str) 
    cr['doc_medkit'] = [TextDocument(text=texte) for texte in cr['TEXTE']]


    # Nettoyage du texte
    regle1 = (r"\n|\t|\xa0", " ")
    regle2 = (r"\s+", " ")
    regexp_replacer = RegexpReplacer(output_label="clean_text", rules=[regle1 , regle2])

    # Tokenisation (découpage d'un textes en unité appelé tokens) en phrase
    sentence_tokenizer = SentenceTokenizer(
        output_label="sentence", keep_punct = True
    )

    # Tokenisation en partie de phrase
    # (lorsque que la phrase comporte plusieurs parties séparer par des conjonctions de cooordinations tel que 'mais', 'et', 'ou', etc)
    syntagma_tokenizer = SyntagmaTokenizer(
        output_label="syntagma",
    )

    # Recherche d'entités
    regexp_matcher = RegexpMatcher(rules=regexp_rules,attrs_to_copy=["section","negation","family"])
    
    # Détection de négation
    neg_rules = [
        NegationDetectorRule(regexp=r"\bpas\s*d[' e]\b"),
        NegationDetectorRule(regexp=r"\bpas\b|VIOLPas"),
        NegationDetectorRule(regexp=r"\bnon\b"),
        NegationDetectorRule(regexp=r"\bni\b"),
        NegationDetectorRule(regexp=r"\bsans\b", exclusion_regexps=[r"\bsans\s*doute\b"]),
        NegationDetectorRule(regexp=r"\bne\s*semble\s*pas"),
        NegationDetectorRule(regexp=r"\bpr.vention\b|\bpr.ventif\b"),
        NegationDetectorRule(regexp=r"recherche"),
        NegationDetectorRule(regexp=r"\bab?sence\b"),
        NegationDetectorRule(regexp=r"\baucune\b"),
        NegationDetectorRule(regexp=r"\btentative|tenter?\b"),
        NegationDetectorRule(regexp=r"\b(fait\s*craindre|faisant\s*craindre)\b"),
        NegationDetectorRule(regexp=r"\binform.e?s?\s*du\s*risque"),
        NegationDetectorRule(regexp=r"\bpourrait\s*.tre\b"),
        NegationDetectorRule(regexp=r"\bau\s*cas\s*o.\b"),
        NegationDetectorRule(regexp=r"\b(suspecter|suspect.e?s?|suspicion)\b"),
        NegationDetectorRule(regexp=r"\b(possibilit.s)\b"),
        NegationDetectorRule(regexp=r"\?"),
    ]
    
    neg_detector = NegationDetector(output_label="negation", rules=neg_rules)
    
    # Détection d'antécédents familiaux
    
    family_detector = FamilyDetector(output_label="family")
    
    # Mise en place de la pipeline
    pipeline = Pipeline(
        steps=[
            PipelineStep(regexp_replacer, input_keys=["full_text"], output_keys=["clean_text"]),
            PipelineStep(sentence_tokenizer, input_keys=["clean_text"], output_keys=["sentence"]),
            PipelineStep(syntagma_tokenizer, input_keys=["sentence"], output_keys=["syntagma"]),
            PipelineStep(neg_detector, input_keys=["syntagma"], output_keys=[]),
            PipelineStep(family_detector, input_keys=["syntagma"], output_keys=[]),
            PipelineStep(regexp_matcher, input_keys=["syntagma"], output_keys=["entities"]),
        ],
        input_keys=["full_text"],
        output_keys=["entities"],
    )
    
    # Lancement de la pipeline et annotation des comptes rendus
    for idx, document in cr.iterrows():
        
        entities = pipeline.run([document['doc_medkit'].raw_segment])
        
        for entity in entities:
            
            document['doc_medkit'].anns.add(entity)
            

def coder_atcd(annotation,table_final,cr):
    
    """ Codage des antécédents : si une annotation est présente, l'antécédent est associé à True, False sinon"""
    
    table_final[annotation] = None    
    for id_pat, data in cr.groupby('ID_PAT') :  
        for idx, ligne in data.iterrows() :
            annot = cr['doc_medkit'].iloc[idx].anns.get(label=annotation)
            if len(annot) > 0 :
                table_final.at[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0],annotation] = True
    return table_final
    

def coder_atcd_non_familiaux(annotation,table_final,cr):
    
    """ Codage de l'infarctus , comme c'est un antécédents souvent mentionné dans la partie antécédents familaiux la détection de contexte familial 
    permet d'exclure les mentions d'infarctus dans la partie antécédents familiaux et de ne conserver que celle lié directement au patient """
    
    table_final[annotation] = None
    for id_pat, data in cr.groupby('ID_PAT') :
        for idx, ligne in data.iterrows() :
            annot = cr['doc_medkit'].iloc[idx].anns.get(label=annotation)
            for j in range (len(annot)):
                if annot[j].attrs.get(label="family")[0].value is False :
                    table_final.loc[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0],annotation] = True
    return table_final
                    
        

def coder_atcd_familiaux(annotation,table_final,cr):
    
    """ Codage d'antécédents familiaux : la détection de contexte familial permet de détecter seulement les mentions d'hémorragie sous arachoidienne 
    ou d'accident vasculaire cérébral que l'entourage familiale du patient aurait eu """
    
    table_final[annotation] = None
    for id_pat, data in cr.groupby('ID_PAT') :
        for idx, ligne in data.iterrows() :
            annot = cr['doc_medkit'].iloc[idx].anns.get(label=annotation)
            for j in range (len(annot)):
                if annot[j].attrs.get(label="family")[0].value is True :
                    table_final.loc[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0],annotation] = True
    return table_final

def coder_evenement(annotation, table_final,cr):
    
    """ Codage des événements se produisant durant le séjour : si le patient présente l'évênement au cours du séjour celui ci sera codé True, False sinon """
    
    table_final[annotation] = None
    for id_pat, data in cr.groupby('ID_PAT') :
        trouver = False
        codage = None
        for idx, ligne in data.iterrows() :
            annot = cr['doc_medkit'].iloc[idx].anns.get(label=annotation)
            for j in range (len(annot)):
                if annot[j].attrs.get(label="negation")[0].value is False :
                    codage = True
                    trouver = True
                elif annot[j].attrs.get(label="negation")[0].value is True and not trouver:
                    codage = False    
           
        table_final.at[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0],annotation] = codage
        
    return table_final


def coder_score_HSA(annotation,table_final,cr):
    
    """ Codage des scores de gravité de l'HSA (score de Fisher, score WFNS), score pris en priorité dans les CR de neurologie (plus fiable),
    en cas d’incohérence, le score le plus élevé, correspondant à l’hémorragie la plus intense, est retenu """
    
    
    normes_score = { "1" : 1, "2" : 2, "3" : 3, "4" : 4, "I" : 1, "II" : 2, "III" : 3, "IV" : 4, "Iv" : 4, "5" : 5, "i" : 1,'V' : 5}
    table_final[annotation] = None
    
    for id_pat, data in cr.groupby('ID_PAT') :
        
        scores_cr_hospi, scores_cr_neuro = [],[]
        trouver,score = None,None
        
        for idx, ligne in data.iterrows() :
            annot = cr['doc_medkit'].iloc[idx].anns.get(label=annotation)
            
            for j in range (len(annot)):
                
                score = int(normes_score[annot[j].text.strip()])
                
                if (ligne["TYPE_DOC"] == "LN:18748-4" or ligne["TYPE_DOC"] == "LN:11488-4" or ligne["TYPE_DOC"] == "LN:34874-8") and score not in scores_cr_neuro:
                    scores_cr_neuro.append(score)
                    trouver = True
                    
                if (ligne["TYPE_DOC"] == 'LN:34112-3' or ligne["TYPE_DOC"] == 'LN:77436-4' or ligne["TYPE_DOC"] == 'LN:11490-0' or  ligne["TYPE_DOC"] == 'LN:34749-2') and not trouver and score not in scores_cr_hospi:
                    scores_cr_hospi.append(score)
                    trouver = False
                    
        if trouver :
            serie = pd.Series(scores_cr_neuro)
            score = serie.max()
            
        if not trouver :
            serie = pd.Series(scores_cr_hospi)
            score = serie.max()
                    
        table_final.loc[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0] ,annotation] = score

    return table_final


def coder_score_glasgow(table_final, cr) :
    
    """ Codage du score de Glasgow : le plus faible, correspondant au plus bas niveau d'inconscience, est récupéré  """    
    
    annotation="glasgow"
    table_final[annotation] = None
    
    for id_pat, data in cr.groupby('ID_PAT') :
        
        scores = []
        
        for idx, ligne in data.iterrows() :
            
            annot = cr['doc_medkit'].iloc[idx].anns.get(label=annotation)
            
            for j in range (len(annot)):
                
                score = int(annot[j].text.strip())
                scores.append(score)
                
        scores = pd.Series(scores)
        score = scores.min()
        
        table_final.loc[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0],annotation] = score
        
    return table_final

def coder_localisation_AIC(table_final,cr,normes_localisations) :
    
    """Codage de la localisation de l'anévrisme intracranien (AIC) rompu : la localisation la plus fréquente est récupérée,
    en cas d'égalité la localisation  la plus probable est retenu"""
  
    for id_pat , data in cr.groupby('ID_PAT') :
    
        localisation_AIC = None
        frequences_localisations = pd.DataFrame({
            "loc" : ['CAROTIDE INTERNE','SYLVIEN', 'ACA', 'COMMUNICANT ANTERIEUR', 'CEREBRALE POSTERIEUR', 'BASILAIRE'],
           "freq" : [ 0, 0, 0, 0, 0, 0] })
        
        for idx, ligne in data.iterrows() :
            
            annot = cr['doc_medkit'].iloc[idx].anns.get(label="localisation_AIC")
    
            for j in range (len(annot)):
    
                for localisation in normes_localisations:
                    
                    if re.search(localisation, annot[j].text,flags=re.IGNORECASE) :
            
                        frequences_localisations.loc[frequences_localisations['loc'] == normes_localisations[localisation] ,'freq'] +=1
    
        localisations = list(frequences_localisations[frequences_localisations['freq'] == frequences_localisations['freq'].max()]['loc'])
        
        # Cas d'égalité : plusieurs localisations ont la fréquence la plus élévé
        if len(localisations)> 1 and frequences_localisations['freq'].max() > 0:
            conditions = [localisations == ['CAROTIDE INTERNE','SYLVIEN'],localisations == ['SYLVIEN', 'COMMUNICANT ANTERIEUR'],
                         localisations == ['COMMUNICANT ANTERIEUR','BASILAIRE'], localisations == ['ACA', 'COMMUNICANT ANTERIEUR'],
                         localisations == ['CAROTIDE INTERNE', 'CEREBRALE POSTERIEUR']]
            choices = ['SYLVIEN','COMMUNICANT ANTERIEUR','COMMUNICANT ANTERIEUR','COMMUNICANT ANTERIEUR','CAROTIDE INTERNE']
            localisation_AIC = str(np.select(conditions, choices))
        
        # Autres cas la localisations la plus fréquente est retenue
        elif frequences_localisations['freq'].max() > 0:
            
            localisation_AIC = frequences_localisations.loc[frequences_localisations[frequences_localisations['freq'] == frequences_localisations['freq'].max()].index[0], 'loc']
             
        table_final.at[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0],'localisation_AIC'] = localisation_AIC
        
    return table_final

def coder_type_traitement_AIC(table_final,cr) : 
    
    """ Codage du type de traitement de l'anévrisme intracranien (AIC) rompu : seul les mentions de traitement à la forme affirmative sont retenues,
    une concaténation des traitements mentionnées dans les comptes rendus est réalisé, en cas d'incohérence dans ces combinaisons, 
    le traitement le plus probable est retenu """
        
    
    normes_traitement = {"coïl" : 'SPIRE', "coil" : 'SPIRE', "spire" : 'SPIRE', "clipage" : 'CLIP', r"clip" : 'CLIP', "diverteur de flux" : "STENT",
                  "flow diverter" : "STENT", "stenting" : "STENT", "stent" : "STENT", "flow-diverter" : "STENT", "web" : "WEB", "remodeling" : 'REMODELING'}

    table_final[['SPIRE','CLIP','STENT','WEB','REMODELING']] = None
    
    for id_pat, data in cr.groupby('ID_PAT') :
        
        for idx, ligne in data.iterrows() :
            
            annot = cr['doc_medkit'].iloc[idx].anns.get(label="type_traitement_AIC") 
            
            for j in range (len(annot)):
    
                traiement = annot[j].text.strip().lower()
                
                # Detection de négation
                if annot[j].attrs[0].value is False :
                    
                    table_final.at[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0], normes_traitement[traiement]] = normes_traitement[traiement]
    
    # Concaténation des traitements mentionnées
    table_final[['SPIRE','CLIP','STENT','WEB','REMODELING']] = table_final[['SPIRE','CLIP','STENT','WEB','REMODELING']].replace(np.nan,'')
    table_final['type_traitement_AIC']  = table_final['SPIRE'] + ' + ' + table_final['CLIP'] + ' + ' + table_final['STENT'] + ' + ' + table_final['WEB'] + ' + ' + table_final['REMODELING']
    table_final['type_traitement_AIC'] = table_final['type_traitement_AIC'].str.replace(r'(\s\+\s){2,}', ' + ', regex=True).str.replace(r'^(\s\+\s)+|(\s\+\s)+$', '', regex=True).replace('',np.nan)
    table_final = table_final.drop(columns = ['SPIRE','CLIP','STENT','WEB','REMODELING'])

    # Choix du traitement le plus probable
    conditions = [table_final['type_traitement_AIC'] == 'CLIP + REMODELING',table_final['type_traitement_AIC'] == 'CLIP + STENT',
                  table_final['type_traitement_AIC'] == 'REMODELING', table_final['type_traitement_AIC'] =='SPIRE + CLIP', 
                  table_final['type_traitement_AIC'] == 'SPIRE + CLIP + REMODELING',table_final['type_traitement_AIC'] == 'SPIRE + WEB',
                  table_final['type_traitement_AIC'] == 'SPIRE + STENT + WEB', table_final['type_traitement_AIC'] == 'SPIRE + CLIP + STENT']

    choices = ['CLIP','CLIP','SPIRE + REMODELING','SPIRE', 'SPIRE + REMODELING','WEB','STENT','SPIRE']

    table_final['type_traitement_AIC'] = np.select(conditions, choices, default=table_final['type_traitement_AIC'])

    return table_final

def coder_AIC_instable(table_final,cr,normes_localisations):

        
    """ Codage de l'instabilité de l'anévrisme intracranien (AIC) rompu : la recherche de localisation associés aux vocabulaires d'instabilité permet
    de détecter les cas ou l'instabilité est associé à l'anévrisme rompu, un autre anévrisme, ou à aucune localisation précise """

    table_final["AIC_instable"] = None
    
    for id_pat, data in cr.groupby('ID_PAT') :
        
        annoter = False
        instable = None
        trouver = False
        
        for idx, ligne in data.iterrows() :
            
            annot = cr['doc_medkit'].iloc[idx].anns.get(label="AIC_instable")
            for j in range (len(annot)):
                annoter = True
                loc = table_final.loc[table_final['ID_PAT'] == ligne['ID_PAT'],'localisation_AIC'].values[0]
                for regex in normes_localisations :
                     
                    # L'anévrisme rompu est instable 
                    if re.search(regex ,str(annot[j].text),flags=re.IGNORECASE) and normes_localisations[regex] == loc :
                        instable = True
                        trouver = True
                        
                    # Un autre anévrisme est instable 
                    if re.search(regex,str(annot[j].text),flags=re.IGNORECASE) and not trouver and pd.notna(loc):
                        instable = False
                        
        # On ne sait pas quel anévrisme est instable (localisation non précisée)       
        if annoter and pd.isna(instable):
            instable = 'NON DETERMINANT' 
                                  
        table_final.at[table_final[table_final['ID_PAT'] == ligne['ID_PAT']].index[0],'AIC_instable'] = instable

    return table_final

def traitement_texte(con_oracle,patients):
    
    """ Programme principal de traitement textuel : 
        - Extraction des comptes rendus en fonction du type de variable
        - Lancement de la pipeline medkit avec recherche d'entité à l'aide d'expressions régulières
        - Transformation de l'annotation généré en données structurées (codage variables) """
    

    table_final = patients[['ID_PAT']].sort_values('ID_PAT').reset_index(drop=True)
    
    # Pipeline de traitement des antécédents réalisés sur les comptes rendus d'hospitalisation
    cr_hospi_atcd = extraire_cr_hospi_atcd(con_oracle)
    
    rules_atcd = RegexpMatcher.load_rules(Path('expressions_regulieres/atcd.yml'), encoding='utf-8')
    pipeline_medkit(cr = cr_hospi_atcd, regexp_rules = rules_atcd)
    
    liste_atcd =['cholesterol','tabac','alcool','apnee_sommeil','anticoagulant','antiagregant','HTA_ICA','HTA_BETA','HTA_TZD','HTA_ARA','HTA_IEC',
    'hormonal_med','alcool_med','diabete_med','cholesterol_med', 'goutte']
    for atcd in liste_atcd :
        table_final = coder_atcd(atcd,table_final,cr_hospi_atcd)     
    table_final = coder_atcd_non_familiaux("infarctus",table_final,cr_hospi_atcd)
    table_final = coder_atcd_familiaux("ATCD_familiaux",table_final,cr_hospi_atcd)
    
    # Pipeline de traitements des signes cliniques et complications non neurologiques réalisés sur les comptes rendus d'hospitalisation
    cr_hospi_evt = extraire_cr_hospi_evt(con_oracle)
    
    rules_evt = RegexpMatcher.load_rules(Path('expressions_regulieres/complications_non_neurologique_signes_cliniques.yml'), encoding='utf-8')
    pipeline_medkit(cr = cr_hospi_evt, regexp_rules = rules_evt)
    
    for evt in ['cephale','crise','cardiomyopathie_stress','syndrome_perte_sel']:
        table_final = coder_evenement(evt,table_final,cr_hospi_evt) 

    
    # Pipeline de traitements des complications neurologiques réalisés sur les comptes rendus de neurologie et d'hospitalisation
    cr_hospi_neuro = extraire_cr_hospi_neuro(con_oracle)
    
    rules_evt_neuro_scores= RegexpMatcher.load_rules(Path('expressions_regulieres/complications_neurologiques_scores.yml'), encoding='utf-8')
    pipeline_medkit(cr = cr_hospi_neuro, regexp_rules = rules_evt_neuro_scores)
    
    for evt in ['vasospasme','hydrocephalie','DVE_txt','hemorragie_intra_vent','ischemie_cerebrale_retardee']:
        table_final = coder_evenement(evt,table_final,cr_hospi_neuro) 
    table_final = coder_score_HSA("score_fisher",table_final,cr_hospi_neuro)
    table_final['score_fisher'] = table_final['score_fisher'].replace(5,4)
    table_final = coder_score_HSA("score_wfns",table_final,cr_hospi_neuro)
    table_final = coder_score_glasgow(table_final,cr_hospi_neuro)
   
    
   # Pipeline de traitements des caractéristiques de l'anévrisme rompu réalisés sur les comptes rendus de neurologie
    cr_neuro = extraire_cr_neuro(con_oracle)
    
    rules_AIC = RegexpMatcher.load_rules(Path('expressions_regulieres/caracteristiques_AIC.yml'), encoding='utf-8')
    pipeline_medkit(cr = cr_neuro, regexp_rules = rules_AIC)
    
    normes_loc = {r"carotide\s*interne|art.res?\s*ophtalmiques?|communi(qu|c)ante?\s*post.rieure?s?|r.tro.?carotidien(ne)?s?|terminaison\s*carotidien(ne)?s?|termino.?carotidien(ne)?s?|ch?oro.dien(ne)?s?|siphon\s*carotidien(ne)?s?|carotido.?caverneux?se?s?" : 'CAROTIDE INTERNE',
         r"\bACM\b|c.r.brale?s?\s*moyen(ne)?s?|sylvien(ne)?|bifurcation\s*(.{0,10}artere\s*carotide(nne)?\s*interne|ACI)|M1|M2|M3" : 'SYLVIEN',
         r"c.r.brales?\s*ant.rieure?s?|ACA|p.ri.?calleux?se?s?|calloso.?marginale?s?|A1|A2|A3" : "ACA",
         r"communi(c|qu)ante?s?\s*ant.rieure?s?|ACOMA|complexe\s*communi(qu|c)ante?s?\s*ant.rieure?s?" : "COMMUNICANT ANTERIEUR", 
         r"c.r.brale?s?\s*post.rieure?s?|ACP" : "CEREBRALE POSTERIEUR",
         r"basilaire?s?|termino.?basilaire?s?|vert.brale?s?|AICA|PICA|c.r.belleux" : "BASILAIRE"}
    table_final = coder_localisation_AIC(table_final,cr_neuro,normes_loc)
    table_final = coder_type_traitement_AIC(table_final,cr_neuro)
    table_final = coder_AIC_instable(table_final,cr_neuro,normes_loc)
    
    return table_final

def management_table_final(table_final):
    
    table_final.drop('alcool_med', axis=1, inplace =True)
    
    table_final['maladie_goutte'] = table_final['goutte'] | table_final['goutte_med']
    table_final.drop('goutte', axis=1, inplace =True)
    table_final.drop('goutte_med', axis=1, inplace =True)
    
    table_final['score_glasgow'] =  table_final[['glasgow', 'GLASGOW']].min(axis=1)
    table_final.drop('glasgow', axis=1, inplace =True)
    table_final.drop('GLASGOW', axis=1, inplace =True)
    
    table_final['HTA'] = table_final['HTA'] | table_final['HTA_IEC'] | table_final['HTA_ARA'] | table_final['HTA_BETA'] | table_final['HTA_TZD'] | table_final['HTA_ICA']
    table_final['cholesterol'] = table_final['cholesterol'] | table_final['cholesterol_med'] 
    table_final['diabete'] = np.select([table_final['diabete_med'] == True, table_final['diabete'] == True,table_final['diabete'] == False,
                                       pd.isna(table_final['diabete'])],[True,True,False,None])   
    
    table_final['outcomes'] = np.select([table_final['MODE_SORTIE'] == '5',
                                (table_final['MODE_SORTIE'] == '2') | (table_final['MODE_SORTIE'] == '6') | (table_final['SORTIE_REEDUC'] == True),
                                (table_final['MODE_SORTIE'] == '4') & (table_final['SORTIE_REEDUC'] == False) ,
                                pd.isna(table_final['MODE_SORTIE'])],
                               ['DECES','REEDUC_TRANSFERT','DOMICILE', None])
    
    return table_final[['ID_PAT','SEXE','AGE','POIDS','DATE_ENTREE','DATE_SORTIE', 'DUREE', 'MODE_ENTREE','MODE_SORTIE',
           'UF_ENTREE', 'SERVICE_ENTREE', 'UF_SORTIE', 'SERVICE_SORTIE','TYPE_SEJOUR','URGENCES', 'UF_PARCOURUS', 'SERVICES_PARCOURUS', 'NB_SERVICES_PARCOURUS',
           'HTA','HTA_IEC','HTA_ARA','HTA_BETA','HTA_TZD','HTA_ICA','diabete','diabete_med','cholesterol','cholesterol_med',
           'antiagregant','anticoagulant','hormonal_med','tabac','alcool','apnee_sommeil','infarctus','maladie_goutte','ATCD_familiaux',
           'nimodipine', 'paracetamol', 'noradrenaline','milrinone', 'morphine', 'antiepideptique', 'antiepideptique_HSA',
           'diagnostic', 'DVE', 'DVP', 'angioplastie', 'intubation_orotracheale',
           'traitement_AIC', 'fievre', 'PA_normal', 'PA_eleve', 'PA_bas',  'PA_O2_bas', 'NA_normal', 'NA_eleve',
           'NA_bas',  'desaturation_O2', 'anemie', 'glucose_normal', 'glucose_eleve', 'glucose_bas',
              'cephale', 'crise', 'score_fisher', 'score_wfns', 'score_glasgow','localisation_AIC','type_traitement_AIC',
           'AIC_instable','vasospasme','hydrocephalie','DVE_txt','cardiomyopathie_stress','syndrome_perte_sel',
            'hemorragie_intra_vent','ischemie_cerebrale_retardee','outcomes']] 
    
    
if __name__ == "__main__" :

    ## Connexion à la base de données oracle 
    config_file = "U:/config.yaml"    

    if not os.path.exists(config_file):
        raise FileNotFoundError("Le fichier config.yaml n'existe pas")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    db = config["database"]
    
    try:
        dsn_tns = oracledb.makedsn(db["host"], db["port"], db["service_name"])
        con_oracle = oracledb.connect(user=db["user"], password=db["password"], dsn=dsn_tns)
    except oracledb.DatabaseError:
        print('Impossible de se connecter à %s\n', dsn_tns)


    # Execution de l'algorithme
    patients = extraire_patients(con_oracle)
    sejours_consecutifs = recherche_sejours_consecutifs(con_oracle)
    premiers_sejours_HSA = recherche_premiers_sejours_HSA(sejours_consecutifs,con_oracle)
    sejours_MCO = exclure_sejours_non_MCO(premiers_sejours_HSA)
    sejours_aggreges = aggreger_sejours(sejours_MCO,patients)
    charger_sejours(con_oracle,sejours_aggreges,sejours_MCO)

    donnees_structurees = extraire_donnees_structurees(sejours_aggreges,con_oracle)
    hta_diabete = extraire_atcd_avec_transformers(con_oracle)
    donnees_textes = traitement_texte(con, patients)

    table_final = patients
    table_a_regrouper = [sejours_aggreges,donnees_structurees,hta_diabete,donnees_textes]
    for table in table_a_regrouper: 
        table_final = table_final.merge(table, on='ID_PAT', how ='left')
    table_final = management_table_final(table_final)
    table_final.to_excel('table_final.xlsx',index=False)
