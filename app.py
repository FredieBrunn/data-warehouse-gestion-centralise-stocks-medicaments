import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
from faker import Faker
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Système de Gestion des Stocks de Médicaments - Cameroun",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Faker for generating sample data
fake = Faker()

# Function to generate synthetic data for each dimension and fact table
def generate_medicaments(n=100):
    categories = ['Antibiotiques', 'Antipaludéens', 'Antalgiques', 'Antiviraux', 
                  'Anti-inflammatoires', 'Vaccins', 'Sérums', 'Antiseptiques', 'Antihistaminiques', 'Vitamines']
    formes = ['Comprimé', 'Sirop', 'Injectable', 'Pommade', 'Suspension', 
              'Capsule', 'Gouttes', 'Poudre', 'Suppositoire', 'Spray']
    fabricants = ['Pharma Cameroun', 'AfriMed', 'SanofiPasteur', 'Merck', 'GlaxoSmithKline',
                 'Pfizer', 'Novartis', 'Roche', 'Bayer', 'Johnson&Johnson']
    
    data = []
    for i in range(1, n+1):
        data.append({
            'id_medicament': i,
            'nom': fake.word().capitalize() + fake.random_element(['-A', '-B', '-C', '-D', '']),
            'nom_dci': fake.word().capitalize() + fake.random_element(['xin', 'cin', 'mol', 'dol', 'zol']),
            'forme': fake.random_element(formes),
            'dosage': fake.random_element(['100mg', '250mg', '500mg', '1g', '5mg', '10mg', '20ml', '50ml', '125mg', '30mg']),
            'catégorie': fake.random_element(categories),
            'fabricant': fake.random_element(fabricants)
        })
    return pd.DataFrame(data)

def generate_centres(n=50):
    types = ['Hôpital de District', 'Centre de Santé', 'Clinique', 'Dispensaire', 'Centre Médical']
    regions = ['Adamaoua', 'Centre', 'Est', 'Extrême-Nord', 'Littoral', 'Nord', 'Nord-Ouest', 'Ouest', 'Sud', 'Sud-Ouest']
    
    region_deps = {
        'Adamaoua': ['Vina', 'Mbéré', 'Djerem', 'Mayo-Banyo', 'Faro-et-Déo'],
        'Centre': ['Mfoundi', 'Nyong-et-Kéllé', 'Lekié', 'Mbam-et-Inoubou', 'Haute-Sanaga'],
        'Est': ['Lom-et-Djerem', 'Kadey', 'Haut-Nyong', 'Boumba-et-Ngoko'],
        'Extrême-Nord': ['Diamaré', 'Mayo-Sava', 'Mayo-Tsanaga', 'Logone-et-Chari', 'Mayo-Danay', 'Mayo-Kani'],
        'Littoral': ['Wouri', 'Sanaga-Maritime', 'Moungo', 'Nkam'],
        'Nord': ['Bénoué', 'Mayo-Louti', 'Mayo-Rey', 'Faro'],
        'Nord-Ouest': ['Mezam', 'Momo', 'Bui', 'Donga-Mantung', 'Ngo-Ketunjia'],
        'Ouest': ['Mifi', 'Menoua', 'Bamboutos', 'Ndé', 'Haut-Nkam', 'Noun', 'Koung-Khi'],
        'Sud': ['Océan', 'Mvila', 'Dja-et-Lobo', 'Vallée-du-Ntem'],
        'Sud-Ouest': ['Fako', 'Meme', 'Ndian', 'Manyu', 'Lebialem', 'Kupe-Muanenguba']
    }
    
    region_coords = {
        'Adamaoua': (7.3, 13.5),
        'Centre': (4.6, 11.5),
        'Est': (4.5, 14.0),
        'Extrême-Nord': (11.0, 14.5),
        'Littoral': (4.2, 10.1),
        'Nord': (9.0, 13.5),
        'Nord-Ouest': (6.5, 10.5),
        'Ouest': (5.5, 10.5),
        'Sud': (2.7, 11.0),
        'Sud-Ouest': (4.5, 9.3)
    }
    
    niveaux = [1, 2, 3]
    
    data = []
    for i in range(1, n+1):
        region = fake.random_element(regions)
        region_lat, region_lon = region_coords[region]
        lat_offset = (fake.random.random() - 0.5) * 1.0
        lon_offset = (fake.random.random() - 0.5) * 1.0
        
        data.append({
            'id_centre': i,
            'nom': fake.random_element(['Centre de Santé ', 'Hôpital ', 'Dispensaire ', 'Clinique ', 'Poste de Santé ']) + fake.city(),
            'type': fake.random_element(types),
            'région': region,
            'département': fake.random_element(region_deps[region]),
            'latitude': region_lat + lat_offset,
            'longitude': region_lon + lon_offset,
            'niveau': fake.random_element(niveaux)
        })
    return pd.DataFrame(data)
def generate_utilisateurs(n=150, centres_df=None):
    postes = ['Médecin', 'Infirmier', 'Pharmacien', 'Administrateur', 'Logisticien', 'Technicien', 'Gestionnaire']
    niveaux_acces = [1, 2, 3, 4, 5]  # 1: basic, 5: admin
    
    data = []
    for i in range(1, n+1):
        if centres_df is not None:
            id_centre = fake.random_element(centres_df['id_centre'].tolist())
        else:
            id_centre = fake.random_int(min=1, max=50)
            
        data.append({
            'id_utilisateur': i,
            'nom': fake.name(),
            'poste': fake.random_element(postes),
            'id_centre': id_centre,
            'niveau_acces': fake.random_element(niveaux_acces)
        })
    return pd.DataFrame(data)

def generate_temps(start_date='2023-01-01', periods=365):
    start = pd.to_datetime(start_date)
    dates = [start + pd.Timedelta(days=i) for i in range(periods)]
    
    data = []
    for i, date in enumerate(dates, 1):
        data.append({
            'id_temps': i,
            'jour': date.day,
            'semaine': date.isocalendar()[1],
            'mois': date.month,
            'trimestre': (date.month-1)//3 + 1,
            'année': date.year,
            'date_complete': date
        })
    return pd.DataFrame(data)

def generate_fournisseurs(n=20):
    pays_origine = ['Cameroun', 'France', 'Inde', 'Chine', 'Afrique du Sud', 'Maroc', 
                   'Sénégal', 'Kenya', 'Nigéria', 'Belgique', 'Allemagne', 'Suisse']
    types = ['Local', 'International', 'Gouvernemental', 'ONG', 'Grossiste', 'Fabricant direct']
    
    data = []
    for i in range(1, n+1):
        data.append({
            'id_fournisseur': i,
            'nom': fake.company(),
            'type': fake.random_element(types),
            'pays_origine': fake.random_element(pays_origine)
        })
    return pd.DataFrame(data)

def generate_stock_facts(medicaments_df, centres_df, temps_df, fournisseurs_df=None):
    data = []
    fact_id = 1
    
    medicament_ids = medicaments_df['id_medicament'].tolist()
    centre_ids = centres_df['id_centre'].tolist()
    temps_ids = temps_df['id_temps'].tolist()
    
    # Create a smaller subset of data points for performance
    sample_size = min(2500, len(medicament_ids) * len(centre_ids) // 40)
    
    for _ in range(sample_size):
        medicament_id = fake.random_element(medicament_ids)
        centre_id = fake.random_element(centre_ids)
        temps_id = fake.random_element(temps_ids)
        
        if fournisseurs_df is not None:
            fournisseur_id = fake.random_element(fournisseurs_df['id_fournisseur'].tolist())
        else:
            fournisseur_id = None
            
        # Generate realistic stock data
        quantite_initiale = fake.random_int(min=0, max=1000)
        quantite_entree = fake.random_int(min=0, max=500)
        quantite_sortie = fake.random_int(min=0, max=min(quantite_initiale + quantite_entree, 300))
        quantite_finale = quantite_initiale + quantite_entree - quantite_sortie
        
        # Calculate price and value data
        prix_unitaire = fake.random_int(min=100, max=10000) / 100  # FCFA
        valeur_stock = quantite_finale * prix_unitaire
        
        # Generate alert levels
        seuil_alerte = fake.random_int(min=20, max=100)
        
        data.append({
            'id_stock': fact_id,
            'id_medicament': medicament_id,
            'id_centre': centre_id,
            'id_temps': temps_id,
            'id_fournisseur': fournisseur_id,
            'quantite_initiale': quantite_initiale,
            'quantite_entree': quantite_entree,
            'quantite_sortie': quantite_sortie,
            'quantite_finale': quantite_finale,
            'prix_unitaire': prix_unitaire,
            'valeur_stock': valeur_stock,
            'seuil_alerte': seuil_alerte,
        })
        fact_id += 1
        
    return pd.DataFrame(data)

def generate_transactions(stock_facts_df, utilisateurs_df, temps_df):
    data = []
    transaction_id = 1
    transaction_types = ['Réception', 'Distribution', 'Ajustement', 'Inventaire', 'Retour', 'Expiration']
    
    # Generate transactions based on stock movements
    sample_size = min(len(stock_facts_df) // 2, 1000)  # Limit sample size for performance
    
    for _ in range(sample_size):
        stock_row = stock_facts_df.sample().iloc[0]
        user = utilisateurs_df.sample().iloc[0]
        temps = temps_df.sample().iloc[0]
        
        quantite = fake.random_int(min=1, max=50)
        transaction_type = fake.random_element(transaction_types)
        
        data.append({
            'id_transaction': transaction_id,
            'id_medicament': stock_row['id_medicament'],
            'id_centre': stock_row['id_centre'],
            'id_utilisateur': user['id_utilisateur'],
            'id_temps': temps['id_temps'],
            'quantite': quantite,
            'type_transaction': transaction_type,
            'commentaire': f"{transaction_type} de {quantite} unités par {user['nom']}"
        })
        transaction_id += 1
        
    return pd.DataFrame(data)

def generate_alerte_stocks(stock_facts_df, centres_df, medicaments_df):
    alerts = []
    
    for _, row in stock_facts_df.iterrows():
        if row['quantite_finale'] <= row['seuil_alerte']:
            medicament_name = medicaments_df[medicaments_df['id_medicament'] == row['id_medicament']]['nom'].iloc[0]
            centre_name = centres_df[centres_df['id_centre'] == row['id_centre']]['nom'].iloc[0]
            region = centres_df[centres_df['id_centre'] == row['id_centre']]['région'].iloc[0]
            
            severity = 'Critique' if row['quantite_finale'] == 0 else 'Moyenne' if row['quantite_finale'] < row['seuil_alerte'] / 2 else 'Faible'
            
            alerts.append({
                'id_medicament': row['id_medicament'],
                'nom_medicament': medicament_name,
                'id_centre': row['id_centre'],
                'nom_centre': centre_name,
                'région': region,
                'quantite_actuelle': row['quantite_finale'],
                'seuil_alerte': row['seuil_alerte'],
                'severite': severity,
                'date_alerte': datetime.now().strftime('%Y-%m-%d')
            })
    
    return pd.DataFrame(alerts)

def generate_epidemiologic_data(centres_df, medicaments_df, temps_df):
    maladies = ['Paludisme', 'Tuberculose', 'VIH/SIDA', 'Choléra', 'Fièvre typhoïde', 
                'Méningite', 'Rougeole', 'Pneumonie', 'Diarrhée', 'Malnutrition']
    
    data = []
    
    for _, centre in centres_df.sample(n=min(20, len(centres_df))).iterrows():
        for maladie in maladies[:5]:  # Limit to 5 diseases per center for simplicity
            for _, temps in temps_df.sample(n=12).iterrows():  # Monthly reports
                data.append({
                    'id_centre': centre['id_centre'],
                    'région': centre['région'],
                    'département': centre['département'],
                    'maladie': maladie,
                    'id_temps': temps['id_temps'],
                    'mois': temps['mois'],
                    'année': temps['année'],
                    'cas_declares': fake.random_int(min=0, max=100),
                    'cas_traites': fake.random_int(min=0, max=80),
                    'medicaments_utilises': [fake.random_element(medicaments_df['id_medicament'].tolist()) for _ in range(3)]
                })
    
    return pd.DataFrame(data)
def generate_campagne_sante(centres_df, medicaments_df, temps_df):
    """Generate sample health campaign data"""
    types_campagne = ['Vaccination', 'Sensibilisation', 'Dépistage', 'Distribution', 'Formation']
    
    data = []
    campaign_id = 1
    
    for _, centre in centres_df.sample(n=min(15, len(centres_df))).iterrows():
        for _ in range(fake.random_int(min=1, max=3)):  # 1-3 campaigns per center
            type_campagne = fake.random_element(types_campagne)
            date_debut = fake.date_between(start_date='-1y', end_date='+3m')
            duree = fake.random_int(min=1, max=14)  # 1-14 days
            date_fin = date_debut + timedelta(days=duree)
            
            # Find the closest matching temps_id
            temps_start = temps_df[temps_df['date_complete'].dt.date >= date_debut]
            if not temps_start.empty:
                temps_id = temps_start.iloc[0]['id_temps']
            else:
                temps_id = fake.random_element(temps_df['id_temps'].tolist())
                
            medicaments_utilises = [fake.random_element(medicaments_df['id_medicament'].tolist()) for _ in range(3)]
            quantites = [fake.random_int(min=10, max=1000) for _ in range(3)]
            
            data.append({
                'id_campagne': campaign_id,
                'nom_campagne': f"Campagne de {type_campagne} - {centre['région']}",
                'id_centre': centre['id_centre'],
                'région': centre['région'],
                'type_campagne': type_campagne,
                'id_temps': temps_id,
                'date_debut': date_debut,
                'date_fin': date_fin,
                'medicaments_utilises': medicaments_utilises,
                'quantites': quantites,
                'population_cible': fake.random_int(min=100, max=10000),
                'population_atteinte': fake.random_int(min=50, max=8000),
                'resultat': fake.random_element(['Excellent', 'Bon', 'Moyen', 'A améliorer'])
            })
            campaign_id += 1
            
    return pd.DataFrame(data)

# ==================== DATA WAREHOUSE CLASS ====================

class MedicationDataWarehouse:
    def __init__(self, load_sample_data=True):
        """Initialize the data warehouse with sample or empty data"""
        if load_sample_data:
            self.load_sample_data()
        else:
            self.init_empty_tables()
    
    def init_empty_tables(self):
        """Initialize empty tables for the warehouse"""
        self.dim_medicament = pd.DataFrame(columns=['id_medicament', 'nom', 'nom_dci', 'forme', 
                                                    'dosage', 'catégorie', 'fabricant'])
        self.dim_centre = pd.DataFrame(columns=['id_centre', 'nom', 'type', 'région', 
                                                'département', 'latitude', 'longitude', 'niveau'])
        self.dim_utilisateur = pd.DataFrame(columns=['id_utilisateur', 'nom', 'poste', 
                                                    'id_centre', 'niveau_acces'])
        self.dim_temps = pd.DataFrame(columns=['id_temps', 'jour', 'semaine', 'mois', 
                                              'trimestre', 'année', 'date_complete'])
        self.dim_fournisseur = pd.DataFrame(columns=['id_fournisseur', 'nom', 'type', 'pays_origine'])
        self.fact_stock = pd.DataFrame(columns=['id_stock', 'id_medicament', 'id_centre', 'id_temps', 
                                               'id_fournisseur', 'quantite_initiale', 'quantite_entree', 
                                               'quantite_sortie', 'quantite_finale', 'prix_unitaire', 
                                               'valeur_stock', 'seuil_alerte'])
        self.fact_transaction = pd.DataFrame(columns=['id_transaction', 'id_medicament', 'id_centre', 
                                                     'id_utilisateur', 'id_temps', 'quantite', 
                                                     'type_transaction', 'commentaire'])
        self.fact_epidemiologie = pd.DataFrame(columns=['id_centre', 'région', 'département', 'maladie', 
                                                       'id_temps', 'mois', 'année', 'cas_declares', 
                                                       'cas_traites', 'medicaments_utilises'])
        self.fact_campagne = pd.DataFrame(columns=['id_campagne', 'nom_campagne', 'id_centre', 'région', 
                                                  'type_campagne', 'id_temps', 'date_debut', 'date_fin', 
                                                  'medicaments_utilises', 'quantites', 'population_cible', 
                                                  'population_atteinte', 'resultat'])
    
    def load_sample_data(self):
        """Load sample data for testing and demonstration"""
        # Generate dimension tables
        self.dim_medicament = generate_medicaments(100)
        self.dim_centre = generate_centres(50)
        self.dim_utilisateur = generate_utilisateurs(150, self.dim_centre)
        self.dim_temps = generate_temps('2023-01-01', 365)
        self.dim_fournisseur = generate_fournisseurs(20)
        
        # Generate fact tables
        self.fact_stock = generate_stock_facts(
            self.dim_medicament, 
            self.dim_centre, 
            self.dim_temps, 
            self.dim_fournisseur
        )
        
        self.fact_transaction = generate_transactions(
            self.fact_stock, 
            self.dim_utilisateur, 
            self.dim_temps
        )
        
        self.fact_epidemiologie = generate_epidemiologic_data(
            self.dim_centre, 
            self.dim_medicament, 
            self.dim_temps
        )
        
        self.fact_campagne = generate_campagne_sante(
            self.dim_centre, 
            self.dim_medicament, 
            self.dim_temps
        )
        
        # Generate alerts
        self.alerts = generate_alerte_stocks(
            self.fact_stock, 
            self.dim_centre, 
            self.dim_medicament
        )
    
    def get_stock_by_region(self):
        """Get stock levels aggregated by region"""
        # Join fact_stock with dim_centre to get region information
        merged_data = pd.merge(
            self.fact_stock, 
            self.dim_centre[['id_centre', 'région']], 
            on='id_centre'
        )
        
        # Aggregate stock by region
        result = merged_data.groupby('région').agg({
            'quantite_finale': 'sum',
            'valeur_stock': 'sum',
            'id_centre': 'nunique'
        }).reset_index()
        
        result.columns = ['Région', 'Quantité totale', 'Valeur totale (FCFA)', 'Nombre de centres']
        return result
    
    def get_stock_alerts(self):
        """Get low stock alerts"""
        return self.alerts.sort_values('severite', ascending=False)
    
    def get_stock_by_medication_category(self):
        """Get stock levels aggregated by medication category"""
        # Join fact_stock with dim_medicament to get category information
        merged_data = pd.merge(
            self.fact_stock, 
            self.dim_medicament[['id_medicament', 'catégorie']], 
            on='id_medicament'
        )
        
        # Aggregate stock by category
        result = merged_data.groupby('catégorie').agg({
            'quantite_finale': 'sum',
            'valeur_stock': 'sum'
        }).reset_index()
        
        result.columns = ['Catégorie', 'Quantité totale', 'Valeur totale (FCFA)']
        return result
    
    def get_stock_history(self, time_period='month', year=2023):
        """Get stock history over time"""
        # Join fact_stock with dim_temps to get time information
        merged_data = pd.merge(
            self.fact_stock, 
            self.dim_temps, 
            on='id_temps'
        )
        
        # Filter by year
        merged_data = merged_data[merged_data['année'] == year]
        
        # Group by the specified time period
        if time_period == 'month':
            result = merged_data.groupby('mois').agg({
                'quantite_finale': 'sum',
                'quantite_entree': 'sum',
                'quantite_sortie': 'sum'
            }).reset_index()
            result.columns = ['Mois', 'Stock final', 'Entrées', 'Sorties']
            
        elif time_period == 'quarter':
            result = merged_data.groupby('trimestre').agg({
                'quantite_finale': 'sum',
                'quantite_entree': 'sum',
                'quantite_sortie': 'sum'
            }).reset_index()
            result.columns = ['Trimestre', 'Stock final', 'Entrées', 'Sorties']
            
        else:  # week
            result = merged_data.groupby('semaine').agg({
                'quantite_finale': 'sum',
                'quantite_entree': 'sum',
                'quantite_sortie': 'sum'
            }).reset_index()
            result.columns = ['Semaine', 'Stock final', 'Entrées', 'Sorties']
            
        return result
    
    def query_data(self, query_type, filters=None):
        """General query function for the data warehouse"""
        if filters is None:
            filters = {}
            
        if query_type == 'stock_by_region':
            return self.get_stock_by_region()
        
        elif query_type == 'stock_alerts':
            return self.get_stock_alerts()
        
        elif query_type == 'stock_by_category':
            return self.get_stock_by_medication_category()
        
        elif query_type == 'stock_history':
            time_period = filters.get('time_period', 'month')
            year = filters.get('year', 2023)
            return self.get_stock_history(time_period, year)
        
        elif query_type == 'medication_details':
            if 'id_medicament' in filters:
                return self.dim_medicament[self.dim_medicament['id_medicament'] == filters['id_medicament']]
            else:
                return self.dim_medicament
        
        elif query_type == 'center_details':
            if 'id_centre' in filters:
                return self.dim_centre[self.dim_centre['id_centre'] == filters['id_centre']]
            else:
                return self.dim_centre
        
        elif query_type == 'epidemiologic_data':
            return self.fact_epidemiologie
        
        elif query_type == 'campaign_data':
            return self.fact_campagne
        
        else:
            return pd.DataFrame()  # Empty dataframe for invalid queries
# ==================== UI FUNCTIONS ====================

def create_download_link(df, filename="data.csv"):
    """Create a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger les données (CSV)</a>'
    return href

def map_severity_to_color(severity):
    """Map severity levels to colors for visual indicators"""
    if severity == 'Critique':
        return 'red'
    elif severity == 'Moyenne':
        return 'orange'
    else:  # 'Faible'
        return 'yellow'

# ==================== STREAMLIT APP DISPLAY FUNCTIONS ====================

def display_home_page():
    st.title("Système de Gestion Centralisée des Stocks de Médicaments")
    st.subheader("Zones Rurales du Cameroun")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Bienvenue dans votre plateforme de gestion des stocks de médicaments
        
        Ce système permet de gérer de manière centralisée les stocks de médicaments dans les zones rurales du Cameroun.
        
        #### Fonctionnalités principales:
        - Visualisation des stocks par région
        - Alertes pour les ruptures de stock
        - Rapports périodiques sur l'état des stocks
        - Analyse épidémiologique
        - Suivi des campagnes de santé
        - Interface d'interrogation de l'entrepôt de données
        
        Utilisez le menu latéral pour naviguer entre les différentes fonctionnalités.
        """)
        
        st.info("""
        **Note:** Les données présentées sont générées aléatoirement à des fins de démonstration.
        Dans un environnement de production, ces données seraient collectées à partir des registres
        des centres de santé, des bases de données locales, des rapports épidémiologiques, des
        fournisseurs pharmaceutiques et des campagnes de santé.
        """)
        
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/4f/Flag_of_Cameroon.svg", 
                 caption="République du Cameroun")
        
        st.markdown("""
        ### Structure de l'entrepôt de données
        
        L'entrepôt de données est organisé selon les dimensions suivantes :
        - **Dim_Medicament** : informations sur les médicaments
        - **Dim_Centre** : informations sur les centres de santé
        - **Dim_Utilisateur** : informations sur les utilisateurs
        - **Dim_Temps** : dimension temporelle
        - **Dim_Fournisseur** : informations sur les fournisseurs
        """)

def display_dashboard(dw):
    st.title("Tableau de Bord")
    
    # Display key metrics
    st.subheader("Indicateurs clés")
    
    # Calculate key metrics
    total_stock = dw.fact_stock['quantite_finale'].sum()
    total_value = dw.fact_stock['valeur_stock'].sum()
    total_centers = dw.dim_centre.shape[0]
    total_medications = dw.dim_medicament.shape[0]
    critical_alerts = len(dw.alerts[dw.alerts['severite'] == 'Critique'])
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Stock", value=f"{total_stock:,.0f} unités")
    with col2:
        st.metric(label="Valeur Totale", value=f"{total_value:,.0f} FCFA")
    with col3:
        st.metric(label="Nombre de Centres", value=f"{total_centers}")
    with col4:
        st.metric(label="Alertes Critiques", value=f"{critical_alerts}", delta="attention" if critical_alerts > 0 else None)
    
    # Display stock by region
    st.subheader("Répartition des stocks par région")
    region_data = dw.get_stock_by_region()
    
    # Create chart for stock by region
    fig_region = px.bar(
        region_data,
        x='Région',
        y='Quantité totale',
        color='Région',
        text_auto=True,
        title="Stocks par région"
    )
    fig_region.update_layout(xaxis_title="Région", yaxis_title="Quantité en stock")
    st.plotly_chart(fig_region, use_container_width=True)
    
    # Display stock by category in a pie chart
    st.subheader("Répartition des stocks par catégorie de médicament")
    category_data = dw.get_stock_by_medication_category()
    
    fig_category = px.pie(
        category_data,
        values='Quantité totale',
        names='Catégorie',
        title="Répartition des stocks par catégorie"
    )
    st.plotly_chart(fig_category, use_container_width=True)
    
    # Display stock alerts
    st.subheader("Alertes de rupture de stock")
    alerts_data = dw.get_stock_alerts()
    
    if len(alerts_data) > 0:
        # Display only the first 10 most critical alerts
        alerts_sample = alerts_data.head(10)
        
        # Display styled dataframe
        st.dataframe(alerts_sample)
        
        st.warning(f"Total des alertes: {len(alerts_data)} (seules les 10 premières sont affichées)")
    else:
        st.success("Aucune alerte de stock actuellement")

def display_regional_stocks(dw):
    st.title("Stocks par Région")
    
    # Get region data
    region_data = dw.get_stock_by_region()
    
    # Add a region selector
    all_regions = sorted(region_data['Région'].unique())
    selected_region = st.selectbox("Sélectionner une région", ["Toutes les régions"] + all_regions)
    
    # Filter data if a specific region is selected
    if selected_region != "Toutes les régions":
        region_data = region_data[region_data['Région'] == selected_region]
    
    # Display region data in a table
    st.subheader("Données de stock par région")
    st.dataframe(region_data)
    
    # Display download link for the data
    st.markdown(create_download_link(region_data, "stocks_par_region.csv"), unsafe_allow_html=True)
    
    # Create a bar chart for the stock quantity
    st.subheader("Quantité de médicaments par région")
    fig_quantity = px.bar(
        region_data,
        x='Région',
        y='Quantité totale',
        color='Région',
        text_auto=True
    )
    fig_quantity.update_layout(xaxis_title="Région", yaxis_title="Quantité en stock")
    st.plotly_chart(fig_quantity, use_container_width=True)
    
    # Create a bar chart for the stock value
    st.subheader("Valeur des stocks par région (FCFA)")
    fig_value = px.bar(
        region_data,
        x='Région',
        y='Valeur totale (FCFA)',
        color='Région',
        text_auto=True
    )
    fig_value.update_layout(xaxis_title="Région", yaxis_title="Valeur du stock (FCFA)")
    st.plotly_chart(fig_value, use_container_width=True)
    
    # Display a map of health centers if a specific region is selected
    if selected_region != "Toutes les régions":
        st.subheader(f"Centres de santé dans la région: {selected_region}")
        
        # Get centers for the selected region
        centers = dw.dim_centre[dw.dim_centre['région'] == selected_region]
        
        # Create a Folium map centered around the average coordinates of centers
        map_center = [centers['latitude'].mean(), centers['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=8)
        
        # Add markers for each center
        for _, center in centers.iterrows():
            # Get stock information for this center
            center_stock = dw.fact_stock[dw.fact_stock['id_centre'] == center['id_centre']]
            total_stock = center_stock['quantite_finale'].sum()
            
            # Create popup content
            popup_content = f"""
            <b>{center['nom']}</b><br>
            Type: {center['type']}<br>
            Département: {center['département']}<br>
            Stock total: {total_stock} unités
            """
            
            # Add marker to map
            folium.Marker(
                location=[center['latitude'], center['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=center['nom'],
                icon=folium.Icon(color='green', icon='hospital', prefix='fa')
            ).add_to(m)
        
        # Display the map
        folium_static(m)
def display_stock_alerts(dw):
    st.title("Alertes de Stock")
    
    # Get alerts data
    alerts_data = dw.get_stock_alerts()
    
    # Display summary metrics
    total_alerts = len(alerts_data)
    critical_alerts = len(alerts_data[alerts_data['severite'] == 'Critique'])
    moderate_alerts = len(alerts_data[alerts_data['severite'] == 'Moyenne'])
    low_alerts = len(alerts_data[alerts_data['severite'] == 'Faible'])
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total des Alertes", value=total_alerts)
    with col2:
        st.metric(label="Alertes Critiques", value=critical_alerts, delta="attention" if critical_alerts > 0 else None)
    with col3:
        st.metric(label="Alertes Moyennes", value=moderate_alerts)
    with col4:
        st.metric(label="Alertes Faibles", value=low_alerts)
    
    # Add filters
    st.subheader("Filtrer les alertes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique regions with alerts
        regions = sorted(alerts_data['région'].unique())
        selected_region = st.selectbox("Région", ["Toutes"] + list(regions))
    
    with col2:
        # Filter by severity
        severities = ['Critique', 'Moyenne', 'Faible']
        selected_severity = st.multiselect("Sévérité", severities, default=severities)
    
    # Apply filters
    filtered_alerts = alerts_data
    if selected_region != "Toutes":
        filtered_alerts = filtered_alerts[filtered_alerts['région'] == selected_region]
    
    if selected_severity:
        filtered_alerts = filtered_alerts[filtered_alerts['severite'].isin(selected_severity)]
    
    # Display filtered alerts
    st.subheader("Liste des alertes")
    
    if len(filtered_alerts) > 0:
        # Reorder columns for better display
        display_cols = ['nom_medicament', 'nom_centre', 'région', 'quantite_actuelle', 
                        'seuil_alerte', 'severite', 'date_alerte']
        
        # Display styled dataframe
        st.dataframe(filtered_alerts[display_cols])
        
        # Download link
        st.markdown(create_download_link(filtered_alerts, "alertes_stock.csv"), unsafe_allow_html=True)
    else:
        st.info("Aucune alerte ne correspond aux filtres sélectionnés.")
    
    # Display alerts by region in a bar chart
    st.subheader("Alertes par région")
    
    alerts_by_region = alerts_data.groupby(['région', 'severite']).size().reset_index(name='count')
    
    fig_alerts_region = px.bar(
        alerts_by_region,
        x='région',
        y='count',
        color='severite',
        title="Nombre d'alertes par région et sévérité",
        barmode='stack'
    )
    fig_alerts_region.update_layout(xaxis_title="Région", yaxis_title="Nombre d'alertes")
    st.plotly_chart(fig_alerts_region, use_container_width=True)

def display_periodic_reports(dw):
    st.title("Rapports Périodiques")
    
    # Time period selector
    st.subheader("Sélectionner la période")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_period = st.radio("Période", ["Semaine", "Mois", "Trimestre"])
        
    with col2:
        year = st.selectbox("Année", [2023])
    
    # Map selection to parameter
    period_param = "week" if time_period == "Semaine" else "month" if time_period == "Mois" else "quarter"
    
    # Get stock history data
    stock_history = dw.get_stock_history(period_param, year)
    
    # Display stock history in a line chart
    st.subheader(f"Évolution des stocks par {time_period.lower()}")
    
    fig = go.Figure()
    
    # Add lines for stock, entries, and exits
    fig.add_trace(go.Scatter(
        x=stock_history.iloc[:, 0],
        y=stock_history['Stock final'],
        mode='lines+markers',
        name='Stock final',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=stock_history.iloc[:, 0],
        y=stock_history['Entrées'],
        mode='lines+markers',
        name='Entrées',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=stock_history.iloc[:, 0],
        y=stock_history['Sorties'],
        mode='lines+markers',
        name='Sorties',
        line=dict(color='red', width=2)
    ))
    
    # Update layout
    x_title = "Semaine" if period_param == "week" else "Mois" if period_param == "month" else "Trimestre"
    fig.update_layout(
        title=f"Évolution des stocks par {time_period.lower()} en {year}",
        xaxis_title=x_title,
        yaxis_title="Quantité",
        legend_title="Légende",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display stock history data in a table
    st.subheader("Données d'évolution des stocks")
    st.dataframe(stock_history)
    
    # Download link for the report data
    st.markdown(create_download_link(stock_history, f"rapport_stock_{period_param}_{year}.csv"), unsafe_allow_html=True)
    
    # Display a summary of stock movements
    st.subheader("Résumé des mouvements de stock")
    
    total_entries = stock_history['Entrées'].sum()
    total_exits = stock_history['Sorties'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total des entrées", value=f"{total_entries:,.0f}")
    with col2:
        st.metric(label="Total des sorties", value=f"{total_exits:,.0f}")
    with col3:
        st.metric(label="Solde net", value=f"{total_entries - total_exits:,.0f}", 
                  delta="positif" if total_entries > total_exits else "négatif")

def display_epidemiologic_analysis(dw):
    st.title("Analyse Épidémiologique")
    
    # Get epidemiologic data
    epi_data = dw.fact_epidemiologie
    
    # Filter options
    st.sidebar.subheader("Filtres")
    
    # Get unique regions and diseases
    regions = sorted(epi_data['région'].unique())
    maladies = sorted(epi_data['maladie'].unique())
    
    # Add filter widgets
    selected_region = st.sidebar.selectbox("Région", ["Toutes"] + list(regions))
    selected_maladie = st.sidebar.selectbox("Maladie", ["Toutes"] + list(maladies))
    
    # Apply filters
    filtered_data = epi_data
    if selected_region != "Toutes":
        filtered_data = filtered_data[filtered_data['région'] == selected_region]
    if selected_maladie != "Toutes":
        filtered_data = filtered_data[filtered_data['maladie'] == selected_maladie]
    
    # Display overview metrics
    st.subheader("Aperçu épidémiologique")
    
    total_cases = filtered_data['cas_declares'].sum()
    total_treated = filtered_data['cas_traites'].sum()
    treatment_rate = (total_treated / total_cases * 100) if total_cases > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Cas déclarés", value=f"{total_cases:,.0f}")
    with col2:
        st.metric(label="Cas traités", value=f"{total_treated:,.0f}")
    with col3:
        st.metric(label="Taux de traitement", value=f"{treatment_rate:.1f}%")
    
    # Disease distribution chart
    st.subheader("Répartition des maladies")
    
    disease_data = filtered_data.groupby('maladie').agg({
        'cas_declares': 'sum',
        'cas_traites': 'sum'
    }).reset_index()
    
    fig_disease = px.bar(
        disease_data,
        x='maladie',
        y=['cas_declares', 'cas_traites'],
        barmode='group',
        title="Cas déclarés et traités par maladie"
    )
    fig_disease.update_layout(xaxis_title="Maladie", yaxis_title="Nombre de cas")
    st.plotly_chart(fig_disease, use_container_width=True)
    
    # Regional distribution chart (if all regions selected)
    if selected_region == "Toutes":
        st.subheader("Répartition géographique")
        
        region_data = filtered_data.groupby('région').agg({
            'cas_declares': 'sum'
        }).reset_index()
        
        fig_region = px.choropleth(
            region_data,
            locations='région',
            color='cas_declares',
            title="Cas par région",
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig_region, use_container_width=True)
    
    # Time trend analysis
    st.subheader("Évolution temporelle")
    
    time_data = filtered_data.groupby(['mois', 'année']).agg({
        'cas_declares': 'sum',
        'cas_traites': 'sum'
    }).reset_index()
    
    time_data['period'] = time_data['année'].astype(str) + '-' + time_data['mois'].astype(str).str.zfill(2)
    time_data = time_data.sort_values('period')
    
    fig_time = px.line(
        time_data,
        x='period',
        y=['cas_declares', 'cas_traites'],
        title="Évolution des cas au fil du temps"
    )
    fig_time.update_layout(xaxis_title="Période", yaxis_title="Nombre de cas")
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Display detailed data
    st.subheader("Données détaillées")
    st.dataframe(filtered_data)
    st.markdown(create_download_link(filtered_data, "donnees_epidemiologiques.csv"), unsafe_allow_html=True)

def display_health_campaigns(dw):
    st.title("Campagnes de Santé")
    
    # Get campaign data
    campaign_data = dw.fact_campagne
    
    # Display summary metrics
    total_campaigns = len(campaign_data)
    total_population_target = campaign_data['population_cible'].sum()
    total_population_reached = campaign_data['population_atteinte'].sum()
    coverage_rate = (total_population_reached / total_population_target * 100) if total_population_target > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Campagnes", value=total_campaigns)
    with col2:
        st.metric(label="Population cible", value=f"{total_population_target:,.0f}")
    with col3:
        st.metric(label="Population atteinte", value=f"{total_population_reached:,.0f}")
    with col4:
        st.metric(label="Taux de couverture", value=f"{coverage_rate:.1f}%")
    
    # Filter options
    st.sidebar.subheader("Filtres")
    
    # Get unique campaign types and regions
    campaign_types = sorted(campaign_data['type_campagne'].unique())
    regions = sorted(campaign_data['région'].unique())
    
    # Add filter widgets
    selected_type = st.sidebar.selectbox("Type de campagne", ["Tous"] + list(campaign_types))
    selected_region = st.sidebar.selectbox("Région", ["Toutes"] + list(regions))
    
    # Apply filters
    filtered_data = campaign_data
    if selected_type != "Tous":
        filtered_data = filtered_data[filtered_data['type_campagne'] == selected_type]
    if selected_region != "Toutes":
        filtered_data = filtered_data[filtered_data['région'] == selected_region]
    
    # Display campaign types distribution
    st.subheader("Types de campagnes")
    
    type_data = filtered_data.groupby('type_campagne').agg({
        'id_campagne': 'count',
        'population_cible': 'sum',
        'population_atteinte': 'sum'
    }).reset_index()
    
    type_data.columns = ['Type de campagne', 'Nombre de campagnes', 'Population cible', 'Population atteinte']
    
    fig_type = px.bar(
        type_data,
        x='Type de campagne',
        y='Nombre de campagnes',
        color='Type de campagne',
        text_auto=True,
        title="Répartition des campagnes par type"
    )
    st.plotly_chart(fig_type, use_container_width=True)
    
    # Display regional distribution
    st.subheader("Répartition géographique des campagnes")
    
    region_data = filtered_data.groupby('région').agg({
        'id_campagne': 'count'
    }).reset_index()
    
    region_data.columns = ['Région', 'Nombre de campagnes']
    
    fig_region = px.bar(
        region_data,
        x='Région',
        y='Nombre de campagnes',
        color='Région',
        text_auto=True,
        title="Campagnes par région"
    )
    st.plotly_chart(fig_region, use_container_width=True)
    
    # Display coverage rate by campaign
    st.subheader("Taux de couverture par campagne")
    
    filtered_data['taux_couverture'] = (filtered_data['population_atteinte'] / filtered_data['population_cible'] * 100).round(1)
    
    fig_coverage = px.bar(
        filtered_data,
        x='nom_campagne',
        y='taux_couverture',
        color='type_campagne',
        text_auto=True,
        title="Taux de couverture par campagne (%)",
        height=500
    )
    fig_coverage.update_layout(xaxis_title="Campagne", yaxis_title="Taux de couverture (%)")
    st.plotly_chart(fig_coverage, use_container_width=True)
    
    # Display detailed campaign data
    st.subheader("Détails des campagnes")
    
    display_cols = ['nom_campagne', 'région', 'type_campagne', 'date_debut', 'date_fin', 
                    'population_cible', 'population_atteinte', 'taux_couverture', 'resultat']
    
    st.dataframe(filtered_data[display_cols])
    st.markdown(create_download_link(filtered_data, "campagnes_sante.csv"), unsafe_allow_html=True)

def display_data_query_interface(dw):
    st.title("Interrogation de l'Entrepôt de Données")
    
    # Select query type
    query_types = {
        "stock_by_region": "Stocks par région",
        "stock_by_category": "Stocks par catégorie de médicament",
        "stock_history": "Historique des stocks",
        "stock_alerts": "Alertes de stock",
        "medication_details": "Détails des médicaments",
        "center_details": "Détails des centres de santé",
        "epidemiologic_data": "Données épidémiologiques",
        "campaign_data": "Données des campagnes de santé"
    }
    
    selected_query = st.selectbox("Type de requête", list(query_types.keys()), format_func=lambda x: query_types[x])
    
    # Initialize filters
    filters = {}
    
    # Show appropriate filter options based on query type
    if selected_query == "stock_history":
        col1, col2 = st.columns(2)
        with col1:
            time_period = st.radio("Période", ["Semaine", "Mois", "Trimestre"])
            filters["time_period"] = "week" if time_period == "Semaine" else "month" if time_period == "Mois" else "quarter"
        with col2:
            filters["year"] = st.selectbox("Année", [2023])
            
    elif selected_query == "medication_details":
        medication_list = dw.dim_medicament[['id_medicament', 'nom']].values.tolist()
        medication_options = ["Tous"] + [f"{id_} - {name}" for id_, name in medication_list]
        selected_medication = st.selectbox("Médicament", medication_options)
        
        if selected_medication != "Tous":
            filters["id_medicament"] = int(selected_medication.split(" - ")[0])
            
    elif selected_query == "center_details":
        center_list = dw.dim_centre[['id_centre', 'nom']].values.tolist()
        center_options = ["Tous"] + [f"{id_} - {name}" for id_, name in center_list]
        selected_center = st.selectbox("Centre", center_options)
        
        if selected_center != "Tous":
            filters["id_centre"] = int(selected_center.split(" - ")[0])
    
    # Execute query
    result = dw.query_data(selected_query, filters)
    
    # Display results
    st.subheader("Résultats de la requête")
    
    if not result.empty:
        st.dataframe(result)
        st.markdown(create_download_link(result, f"resultat_{selected_query}.csv"), unsafe_allow_html=True)
    else:
        st.info("Aucun résultat trouvé pour cette requête.")

def display_data_management(dw):
    st.title("Gestion des Données")
    
    st.info("""
    Cette section permet de gérer les données de l'entrepôt.
    Dans un environnement de production, elle permettrait d'importer, de nettoyer
    et de transformer les données provenant de différentes sources.
    """)
    
    # Data source options
    st.subheader("Sources de données")
    
    st.markdown("""
    Les données de cet entrepôt proviennent de plusieurs sources :
    
    1. **Registres des centres de santé** : Stocks, transactions, utilisateurs
    2. **Rapports épidémiologiques** : Cas de maladies, traitements
    3. **Fournisseurs pharmaceutiques** : Information sur les médicaments, livraisons
    4. **Campagnes de santé** : Données sur les campagnes et leurs résultats
    5. **Bases de données locales** : Informations sur les centres de santé
    
    Pour cette démonstration, les données sont générées aléatoirement.
    """)
    
    # Display sample data from each dimension
    st.subheader("Échantillons des tables")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Médicaments", "Centres", "Utilisateurs", "Temps", "Fournisseurs"])
    
    with tab1:
        st.dataframe(dw.dim_medicament.head(10))
    
    with tab2:
        st.dataframe(dw.dim_centre.head(10))
    
    with tab3:
        st.dataframe(dw.dim_utilisateur.head(10))
    
    with tab4:
        st.dataframe(dw.dim_temps.head(10))
    
    with tab5:
        st.dataframe(dw.dim_fournisseur.head(10))
    
    # Data schema visualization
    st.subheader("Schéma de l'entrepôt de données")
    
    st.markdown("""
    ```
    +-----------------+     +-------------+     +-----------------+
    | Dim_Medicament  |     | Dim_Centre  |     | Dim_Utilisateur |
    +-----------------+     +-------------+     +-----------------+
    | id_medicament   |     | id_centre   |     | id_utilisateur  |
    | nom             |     | nom         |     | nom             |
    | nom_dci         |     | type        |     | poste           |
    | forme           |     | région      |     | id_centre       |
    | dosage          |     | département |     | niveau_acces    |
    | catégorie       |     | latitude    |     +-----------------+
    | fabricant       |     | longitude   |
    +-----------------+     | niveau      |     +---------------+
                           +-------------+     | Dim_Temps     |
    +----------------+                         +---------------+
    | Dim_Fournisseur|                         | id_temps      |
    +----------------+                         | jour          |
    | id_fournisseur |                         | semaine       |
    | nom            |                         | mois          |
    | type           |                         | trimestre     |
    | pays_origine   |                         | année         |
    +----------------+                         | date_complete |
                                               +---------------+
                    +----------------+
                    | Fact_Stock     |
                    +----------------+
                    | id_stock       |
                    | id_medicament  |
                    | id_centre      |
                    | id_temps       |
                    | id_fournisseur |
                    | quantite_*     |
                    | prix_unitaire  |
                    | valeur_stock   |
                    | seuil_alerte   |
                    +----------------+
    ```
    """)
    
    # Reset data option
    if st.button("Réinitialiser les données de démonstration"):
        dw.load_sample_data()
        st.success("Les données ont été réinitialisées !")

def main():
    # Create sidebar for navigation
    st.sidebar.image("./img/data-processing_2980479.png", width=100)
    st.sidebar.title("Navigation")
    
    # Session state initialization
    if 'data_warehouse' not in st.session_state:
        st.session_state.data_warehouse = MedicationDataWarehouse(load_sample_data=True)
    
    dw = st.session_state.data_warehouse
    
    # Main menu options
    menu = st.sidebar.selectbox(
        "Menu Principal",
        ["Accueil", "Tableau de Bord", "Stocks par Région", "Alertes de Stock", 
         "Rapports Périodiques", "Analyse Épidémiologique", "Campagnes de Santé", 
         "Interrogation de l'Entrepôt", "Gestion des Données"]
    )
    
    # Display the selected page
    if menu == "Accueil":
        display_home_page()
    
    elif menu == "Tableau de Bord":
        display_dashboard(dw)
    
    elif menu == "Stocks par Région":
        display_regional_stocks(dw)
    
    elif menu == "Alertes de Stock":
        display_stock_alerts(dw)
    
    elif menu == "Rapports Périodiques":
        display_periodic_reports(dw)
    
    elif menu == "Analyse Épidémiologique":
        display_epidemiologic_analysis(dw)
    
    elif menu == "Campagnes de Santé":
        display_health_campaigns(dw)
    
    elif menu == "Interrogation de l'Entrepôt":
        display_data_query_interface(dw)
    
    elif menu == "Gestion des Données":
        display_data_management(dw)

if __name__ == "__main__":
    main()
