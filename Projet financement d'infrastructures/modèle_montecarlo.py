import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

def run_project_finance_model(oil_price, total_capex=500, debt_share=0.7):
    """
    Exécute le modèle financier déterministe pour un scénario de prix donné.
    
    Args:
        oil_price (float): Prix du baril de pétrole (USD).
        total_capex (float): Dépenses d'investissement totales (M USD).
        debt_share (float): Part de la dette dans le financement (0.0 à 1.0).
        
    Returns:
        float: Le Taux de Rentabilité Interne (TRI/IRR) pour l'actionnaire.
    """
    # 1. Hypothèses Opérationnelles
    years_construction = 2
    years_operation = 15
    total_years = years_construction + years_operation
    
    production_bpd = 10000
    opex_per_barrel = 15.0
    tax_rate = 0.30
    
    # 2. Structure de Financement
    # equity_share = 1 - debt_share (Implicite)
    debt_total = total_capex * debt_share
    equity_total = total_capex * (1 - debt_share)
    interest_rate = 0.07
    
    # Initialisation du DataFrame
    df = pd.DataFrame(index=range(1, total_years + 1))
    
    # 3. Modélisation des Flux
    # Phase
    df['Phase'] = ['Construction' if y <= years_construction else 'Operation' for y in df.index]
    
    # Capex et Tirages (Drawdowns)
    df['Capex'] = np.where(df['Phase'] == 'Construction', total_capex / years_construction, 0)
    df['Drawdown_Debt'] = df['Capex'] * debt_share
    df['Drawdown_Equity'] = df['Capex'] * (1 - debt_share)
    
    # Revenus et Opex
    annual_revenue = (production_bpd * 365 * oil_price) / 1_000_000 
    annual_opex = (production_bpd * 365 * opex_per_barrel) / 1_000_000
    
    df['Revenue'] = np.where(df['Phase'] == 'Operation', annual_revenue, 0)
    df['Opex'] = np.where(df['Phase'] == 'Operation', annual_opex, 0)
    df['EBITDA'] = df['Revenue'] - df['Opex']
    
    # Fiscalité et CFADS
    df['Tax'] = df['EBITDA'] * tax_rate
    df['CFADS'] = df['EBITDA'] - df['Tax']
    
    # 4. Service de la Dette (Calcul itératif)
    debt_balance = []
    interest_payment = []
    principal_payment = []
    
    current_balance = 0
    annual_principal = debt_total / years_operation
    
    for i in df.index:
        phase = df.loc[i, 'Phase']
        interest = current_balance * interest_rate
        interest_payment.append(interest)
        
        if phase == 'Construction':
            current_balance += df.loc[i, 'Drawdown_Debt']
            principal = 0
        else:
            principal = min(current_balance, annual_principal)
            current_balance -= principal
        
        principal_payment.append(principal)
        debt_balance.append(current_balance)
        
    df['Total_Debt_Service'] = np.array(interest_payment) + np.array(principal_payment)
    
    # 5. Flux de Trésorerie Actionnaire (Cash Flow to Equity)
    df['Cash_Flow_Equity'] = np.where(
        df['Phase'] == 'Construction',
        -df['Drawdown_Equity'],
        df['CFADS'] - df['Total_Debt_Service']
    )
    
    # Calcul du TRI (IRR)
    try:
        irr = npf.irr(df['Cash_Flow_Equity'])
    except:
        irr = np.nan
        
    return irr

def run_monte_carlo_simulation(n_simulations=1000, avg_price=45, volatility=15):
    """
    Exécute une simulation de Monte Carlo pour évaluer la distribution du TRI.
    Génère un histogramme des résultats.
    
    Args:
        n_simulations (int): Nombre d'itérations.
        avg_price (float): Prix moyen du baril attendu.
        volatility (float): Écart-type du prix (volatilité).
    """
    print(f"Initialisation de la simulation Monte Carlo ({n_simulations} iterations)...")
    
    # Génération des prix aléatoires (Loi Normale)
    # np.maximum(0, ...) assure que le prix ne soit jamais négatif
    random_prices = np.maximum(0, np.random.normal(avg_price, volatility, n_simulations))
    
    results_irr = []
    
    # Boucle de simulation
    for price in random_prices:
        irr = run_project_finance_model(oil_price=price)
        
        # On ne conserve que les résultats valides
        if not np.isnan(irr):
            results_irr.append(irr * 100) # Conversion en pourcentage
            
    # Conversion en array numpy pour analyse statistique
    results_irr = np.array(results_irr)
    
    # Statistiques clés
    avg_irr = np.mean(results_irr)
    min_irr = np.min(results_irr)
    max_irr = np.max(results_irr)
    
    # Probabilité d'atteindre le seuil de rentabilité (Hurdle Rate > 10%)
    hurdle_rate = 10.0
    prob_success = np.sum(results_irr > hurdle_rate) / len(results_irr) * 100
    
    print("Simulation terminee.")
    print("-" * 30)
    print(f"TRI Moyen :               {avg_irr:.2f}%")
    print(f"TRI Minimum :             {min_irr:.2f}%")
    print(f"TRI Maximum :             {max_irr:.2f}%")
    print(f"Probabilite (TRI > {hurdle_rate}%) : {prob_success:.1f}%")
    print("-" * 30)
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    
    plt.hist(results_irr, bins=50, color='#4F81BD', edgecolor='black', alpha=0.7, label='Distribution TRI')
    plt.axvline(avg_irr, color='red', linestyle='dashed', linewidth=1.5, label=f'Moyenne ({avg_irr:.2f}%)')
    plt.axvline(hurdle_rate, color='green', linewidth=1.5, label=f'Hurdle Rate ({hurdle_rate}%)')
    
    plt.title(f'Distribution des Rendements (Monte Carlo - {n_simulations} Scénarios)', fontsize=12)
    plt.xlabel('Taux de Rentabilité Interne (TRI %)')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_filename = 'monte_carlo_distribution.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Graphique exporte : {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Exécution du script
    run_monte_carlo_simulation(n_simulations=1000, avg_price=45, volatility=15)