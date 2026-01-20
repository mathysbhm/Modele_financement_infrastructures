"""
Project Finance Modeling Tool - Energy Sector (Python)
------------------------------------------------------
Auteur : Mathys Brahmia Ferrier
Description : 
    Modélisation des flux de trésorerie (Cash Flow Waterfall) pour un projet d'infrastructure
    énergétique type "Gaz et pétrole offshore". Le modèle calcule le service de la dette,
    la taxe , et les ratios de couverture (DSCR) selon des scénarios de prix.
    
Sorties :
    - Fichier Excel formaté (Hypothèses + Cash Flows du projet)
    - Dashboard visuel (PNG) avec analyse de sensibilité selon le prix du pétrole
"""


import sys
import subprocess
import importlib.util

# BLOC D'AUTO-INSTALLATION DE xlsxwriter
def install_package(package):
    print(f"🔧 Tentative d'installation automatique de : {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"✅ {package} installé avec succès !")

# Vérifie si xlsxwriter est présent, sinon l'installe
if importlib.util.find_spec("xlsxwriter") is None:
    install_package("xlsxwriter")


import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

def run_project_finance_model(oil_price=70, total_capex=500, debt_share=0.7):
    """
    Modèle de financement de projets pour une plateforme pétrolière (TotalEnergies SPV).
    Unités : Millions USD
    """
    
    # 1. Hypothèses basiques
    years_construction = 2       # Durée construction
    years_operation = 15         # Durée d'exploitation
    total_years = years_construction + years_operation
    
    production_bpd = 10000       # Barils par jour
    opex_per_barrel = 15.0       # Coût d'extraction par baril
    tax_rate = 0.30
    
    # 2. Structure de financement 
    equity_share = 1 - debt_share
    debt_total = total_capex * debt_share
    equity_total = total_capex * equity_share
    interest_rate = 0.07
    
    # Modélisation année par année avec dataframe
    df = pd.DataFrame(index=range(1, total_years + 1))
    
    # 3. Modélisation globale
    
    # A. Phase Construction vs Opération
    df['Phase'] = ['Construction' if y <= years_construction else 'Operation' for y in df.index]
    
    # B. Capex (Réparti uniformément sur la phase construction)
    df['Capex'] = np.where(df['Phase'] == 'Construction', total_capex / years_construction, 0)
    
    # C. Tirage Dette & Equity (Pendant la construction)
    df['Drawdown_Debt'] = df['Capex'] * debt_share
    df['Drawdown_Equity'] = df['Capex'] * equity_share
    
    # D. Revenus & Opex (Seulement en Opération)
    # Revenu annuel = bpd * 365 * prix
    annual_revenue = (production_bpd * 365 * oil_price) / 1_000_000 # En M$
    annual_opex = (production_bpd * 365 * opex_per_barrel) / 1_000_000
    
    df['Revenue'] = np.where(df['Phase'] == 'Operation', annual_revenue, 0)
    df['Opex'] = np.where(df['Phase'] == 'Operation', annual_opex, 0)
    df['EBITDA'] = df['Revenue'] - df['Opex']
    
    # E. Tax (volontairement simplifié sur EBITDA ici)
    df['Tax'] = df['EBITDA'] * tax_rate
    
    # F. Cash Flow Available for Debt Service (CFADS)
    df['CFADS'] = df['EBITDA'] - df['Tax'] - (df['Capex'] * 0) 
    
    # --- 4. Modélisation de dette ---
    debt_balance = []
    interest_payment = []
    principal_payment = []
    
    current_balance = 0
    
    # Supposition d'amortissement linéaire de la dette sur la période d'opération
    annual_principal = debt_total / years_operation
    
    for i in df.index:
        phase = df.loc[i, 'Phase']
        
        # Intérêts
        interest = current_balance * interest_rate
        interest_payment.append(interest)
        
        # Construction : on tire sur la dette
        if phase == 'Construction':
            drawdown = df.loc[i, 'Drawdown_Debt']
            current_balance += drawdown
            principal = 0
            
        # Opération : Remboursement du principal
        else:
            principal = min(current_balance, annual_principal)
            current_balance -= principal
        
        principal_payment.append(principal)
        debt_balance.append(current_balance)
        
    df['Debt_Balance_EoP'] = debt_balance
    df['Interest'] = interest_payment
    df['Principal_Repayment'] = principal_payment
    df['Total_Debt_Service'] = df['Interest'] + df['Principal_Repayment']
    
    # 5. Ratios basiques et retours
    
    # Cash Flow to Equity
    df['Cash_Flow_Equity'] = np.where(df['Phase']=='Construction', 
                                      -df['Drawdown_Equity'], 
                                      df['CFADS'] - df['Total_Debt_Service'])
    
    # DSCR (Debt Service Coverage Ratio)
    df['DSCR'] = np.where(df['Total_Debt_Service'] > 0, 
                          df['CFADS'] / df['Total_Debt_Service'], 
                          0)
    
    # IRR du projet
    irr = npf.irr(df['Cash_Flow_Equity'])
    
    # On retourne aussi les hypothèses pour le résumé Excel
    assumptions = {
        'Prix du Baril ($)': oil_price,
        'Capex Total (M$)': total_capex,
        'Levier (Dette %)': f"{debt_share:.0%}",
        'Taux Intérêt': f"{interest_rate:.1%}",
        'TRI Actionnaire': f"{irr:.2%}"
    }
    
    return df, irr, assumptions

def export_to_excel(df, summary_dict, filename="Modele_Total_Project_Finance.xlsx"):
    """
    Exporte le modèle vers Excel avec un formatage 'Banque'.
    """
    print(f" Génération du fichier Excel : {filename}")
    
    try:
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            
            # ONGLET 1 : EXECUTIVE SUMMARY 
            # Création d'un petit tableau pour les hypothèses
            summary_df = pd.DataFrame(list(summary_dict.items()), columns=['Métrique', 'Valeur'])
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False, startrow=1, startcol=1)
            
            # Formatage Summary
            workbook = writer.book
            worksheet_summary = writer.sheets['Executive Summary']
            
            # Formats
            header_fmt = workbook.add_format({'bold': True, 'bg_color': "#6BAAF7", 'font_color': 'white', 'border': 1})
            cell_fmt = workbook.add_format({'border': 1})
            
            # Appliquer les formats
            worksheet_summary.set_column('B:B', 25) # Largeur colonne Métrique
            worksheet_summary.set_column('C:C', 15) # Largeur colonne Valeur
            
            # Écrire les headers manuellement pour appliquer le style
            worksheet_summary.write('B2', 'Métrique', header_fmt)
            worksheet_summary.write('C2', 'Valeur', header_fmt)
            
            # --- ONGLET 2 : CASH FLOWS ---
            df.to_excel(writer, sheet_name='Cash Flows')
            worksheet_cf = writer.sheets['Cash Flows']
            
            # Formats Financiers
            money_fmt = workbook.add_format({'num_format': '#,##0.0', 'align': 'center'})
            dscr_fmt = workbook.add_format({'num_format': '0.00"x"', 'align': 'center', 'bold': True, 'font_color': '#006100', 'bg_color': '#C6EFCE'}) # Vert si OK
            
            # Largeur des colonnes et format monétaire par défaut
            worksheet_cf.set_column('A:A', 5)   # Index
            worksheet_cf.set_column('B:B', 12)  # Phase
            worksheet_cf.set_column('C:Z', 12, money_fmt) # Le reste en format chiffré
            
            # Format Spécifique pour la colonne DSCR (supposons qu'elle soit la dernière ou presque)
            # On cherche l'index de la colonne DSCR dans le dataframe
            if 'DSCR' in df.columns:
                # +1 car Excel commence à 0 mais la première colonne est l'index
                col_idx = df.columns.get_loc('DSCR') + 1 
                worksheet_cf.set_column(col_idx, col_idx, 12, dscr_fmt)
                
        print(f" Export réussi, Fichier enregistré sous : {filename}")
        
    except Exception as e:
        print(f" Erreur lors de l'export Excel : {e}")
        print("Vérifiez que le fichier n'est pas déjà ouvert.")

# EXÉCUTION DU SCÉNARIO
print(" Lancement de la Simulation ")

# 1. On lance le calcul
df_res, project_irr, assumptions = run_project_finance_model(oil_price=75, total_capex=600, debt_share=0.7)

# 2. Affichage Console
print(f"TRI Actionnaire (Equity IRR): {project_irr:.2%}")
print(f"DSCR Moyen (Opération): {df_res[df_res['Phase']=='Operation']['DSCR'].mean():.2f}x")

def create_dashboard(df):
    print(" Génération des graphiques...")
    
    # On filtre la phase d'operation pour les graphiques, la phase de construction n'a pas d'intérêt visuel ici
    df_op = df[df['Phase'] == 'Operation']
    years = df_op.index
    
    # Création d'une fenêtre avec 3 graphiques (3 lignes, 1 colonne)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    plt.subplots_adjust(hspace=0.4) # Espace entre les graphiques
    
    # GRAPHIQUE 1 : CFADS vs SERVICE DE LA DETTE
    # Barre verte > barre rouge ?
    ax1.bar(years, df_op['CFADS'], label='CFADS (Cash Dispo)', color='#4F81BD', alpha=0.7)
    ax1.bar(years, df_op['Total_Debt_Service'], label='Service Dette', color='#C0504D', alpha=0.9)
    ax1.set_title('Marge de Sécurité : CFADS vs Service de la Dette', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Millions USD')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # GRAPHIQUE 2 : PROFIL DE LA DETTE (Amortissement)
    # Le risque baisse t'il au cours du temps ?
    ax2.fill_between(years, df_op['Debt_Balance_EoP'], color='orange', alpha=0.3)
    ax2.plot(years, df_op['Debt_Balance_EoP'], color='orange', marker='o', label='Dette Restante (BoP)')
    ax2.set_title("Profil d'Amortissement de la Dette", fontsize=12, fontweight='bold')
    ax2.set_ylabel('Millions USD')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # GRAPHIQUE 3 : DSCR (Le respect des Covenants) 
    ax3.plot(years, df_op['DSCR'], color='green', marker='s', linewidth=2, label='DSCR Projet')
    # Ligne rouge de danger (Covenant bancaire classique)
    ax3.axhline(y=1.30, color='red', linestyle='--', linewidth=2, label='Covenant (1.30x)')
    ax3.set_title("Évolution du DSCR (Couverture de Dette)", fontsize=12, fontweight='bold')
    ax3.set_ylabel('Ratio (x)')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Image sauvegardée et affichée
    plt.savefig('Dashboard_Project_Finance.png', dpi=300)
    print(" Graphique sauvegardé : Dashboard_Project_Finance.png")
    
    # Afficher à l'écran
    plt.show()

# 1. Calcul des flux
df_res, project_irr, assumptions = run_project_finance_model(oil_price=75, total_capex=600, debt_share=0.7)

# 2. Dashboard
create_dashboard(df_res) 

# 3. Export Excel
export_to_excel(df_res, assumptions)