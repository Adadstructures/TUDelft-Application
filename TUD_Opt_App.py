import streamlit as st
import numpy as np
import optuna
import pickle
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# Load the pre-trained CatBoost model
best_catboost = pickle.load(open('best_catboost_model.pkl', 'rb'))

# Default unit costs and carbon footprints
unit_costs = {
    'cement': 0.1, 'slag': 0.05, 'flyash': 0.03, 'water': 0.001,
    'superplasticizer': 0.2, 'coarseaggregate': 0.01, 'fineaggregate': 0.02
}
carbon_footprints = {
    'cement': 0.9, 'slag': 0.05, 'flyash': 0.02, 'water': 0.001,
    'superplasticizer': 0.1, 'coarseaggregate': 0.01, 'fineaggregate': 0.015
}
densities = {
    'cement': 3150, 'slag': 2900, 'flyash': 2200,
    'coarseaggregate': 2650, 'fineaggregate': 2650, 'water': 1000
}

# Define constraints
max_water_cement_ratio = 0.5
min_aggregate_ratio = 2.5
volume_constraint = 1.0  # Volume in m³ (1 cubic meter)
volume_tolerance = 0.02  # 2% tolerance
strength_tolerance = 0.05  # 5% tolerance

# Function to calculate total volume
def calculate_volume(cement, slag, flyash, water, coarseaggregate, fineaggregate):
    volume_cement = cement / densities['cement']
    volume_slag = slag / densities['slag']
    volume_flyash = flyash / densities['flyash']
    volume_water = water / densities['water']
    volume_coarseaggregate = coarseaggregate / densities['coarseaggregate']
    volume_fineaggregate = fineaggregate / densities['fineaggregate']
    return volume_cement + volume_slag + volume_flyash + volume_water + volume_coarseaggregate + volume_fineaggregate

# Optuna objective function
def objective(trial):
    cement = trial.suggest_uniform('cement', 100, 500)
    slag = trial.suggest_uniform('slag', 50, 300)
    flyash = trial.suggest_uniform('flyash', 0, 200)
    water = trial.suggest_uniform('water', 140, 210)
    superplasticizer = trial.suggest_uniform('superplasticizer', 0, 10)
    coarseaggregate = trial.suggest_uniform('coarseaggregate', 800, 1600)
    fineaggregate = trial.suggest_uniform('fineaggregate', 600, 1200)
    age = trial.suggest_uniform('age', 7,28)  # Concrete age in days

    total_binder = cement + slag + flyash
    total_aggregate = coarseaggregate + fineaggregate
    water_to_binder_ratio = water / total_binder
    wc_ratio = water / cement
    total_volume = calculate_volume(cement, slag, flyash, water, coarseaggregate, fineaggregate)

    # Predict compressive strength using CatBoost model
    mix = np.array([[cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age]])
    predicted_compressive_strength = best_catboost.predict(mix)[0]

    total_cost = (cement * unit_costs['cement'] + slag * unit_costs['slag'] +
                  flyash * unit_costs['flyash'] + water * unit_costs['water'] +
                  superplasticizer * unit_costs['superplasticizer'] +
                  coarseaggregate * unit_costs['coarseaggregate'] +
                  fineaggregate * unit_costs['fineaggregate'])

    total_carbon_footprint = (cement * carbon_footprints['cement'] + slag * carbon_footprints['slag'] +
                              flyash * carbon_footprints['flyash'] + water * carbon_footprints['water'] +
                              superplasticizer * carbon_footprints['superplasticizer'] +
                              coarseaggregate * carbon_footprints['coarseaggregate'] +
                              fineaggregate * carbon_footprints['fineaggregate'])

    penalty = 0
    if wc_ratio > max_water_cement_ratio:
        penalty += 1e6 * (wc_ratio - max_water_cement_ratio)
    aggregate_ratio = total_aggregate / total_binder
    if aggregate_ratio < min_aggregate_ratio:
        penalty += 1e6 * (min_aggregate_ratio - aggregate_ratio)
    if not np.isclose(total_volume, volume_constraint, rtol=volume_tolerance):
        penalty += 1e6 * abs(total_volume - volume_constraint)

    lower_bound = target_compressive_strength * (1 - strength_tolerance)
    upper_bound = target_compressive_strength * (1 + strength_tolerance)

    if not (lower_bound <= predicted_compressive_strength <= upper_bound):
        penalty += 1e6 * abs(predicted_compressive_strength - target_compressive_strength)
        
    if total_carbon_footprint > carbon_footprint_constraint:
        penalty += 1e6 * (total_carbon_footprint - carbon_footprint_constraint)

    return total_cost + total_carbon_footprint + penalty

# Streamlit App
st.title("Concrete Mix Optimizer")

# User input for unit costs
st.sidebar.header("Unit Costs")
st.sidebar.subheader("Costs per kilogram")
unit_costs['cement'] = st.sidebar.number_input('Cement cost ($/kg)', value=unit_costs['cement'])
unit_costs['slag'] = st.sidebar.number_input('Slag cost ($/kg)', value=unit_costs['slag'])
unit_costs['flyash'] = st.sidebar.number_input('Flyash cost ($/kg)', value=unit_costs['flyash'])
unit_costs['water'] = st.sidebar.number_input('Water cost ($/kg)', value=unit_costs['water'])
unit_costs['superplasticizer'] = st.sidebar.number_input('Superplasticizer cost ($/kg)', value=unit_costs['superplasticizer'])
unit_costs['coarseaggregate'] = st.sidebar.number_input('Coarse aggregate cost ($/kg)', value=unit_costs['coarseaggregate'])
unit_costs['fineaggregate'] = st.sidebar.number_input('Fine aggregate cost ($/kg)', value=unit_costs['fineaggregate'])

# User input for carbon footprints
st.sidebar.header("Carbon Footprints")
st.sidebar.subheader("Footprint per kilogram (kg CO2-eq)")
carbon_footprints['cement'] = st.sidebar.number_input('Cement carbon footprint (kg CO2-eq/kg)', value=carbon_footprints['cement'])
carbon_footprints['slag'] = st.sidebar.number_input('Slag carbon footprint (kg CO2-eq/kg)', value=carbon_footprints['slag'])
carbon_footprints['flyash'] = st.sidebar.number_input('Flyash carbon footprint (kg CO2-eq/kg)', value=carbon_footprints['flyash'])
carbon_footprints['water'] = st.sidebar.number_input('Water carbon footprint (kg CO2-eq/kg)', value=carbon_footprints['water'])
carbon_footprints['superplasticizer'] = st.sidebar.number_input('Superplasticizer carbon footprint (kg CO2-eq/kg)', value=carbon_footprints['superplasticizer'])
carbon_footprints['coarseaggregate'] = st.sidebar.number_input('Coarse aggregate carbon footprint (kg CO2-eq/kg)', value=carbon_footprints['coarseaggregate'])
carbon_footprints['fineaggregate'] = st.sidebar.number_input('Fine aggregate carbon footprint (kg CO2-eq/kg)', value=carbon_footprints['fineaggregate'])

# User input for densities
st.sidebar.header("Unit Densities")
st.sidebar.subheader("Density in kg/m³")
densities['cement'] = st.sidebar.number_input('Cement density (kg/m³)', value=densities['cement'])
densities['slag'] = st.sidebar.number_input('Slag density (kg/m³)', value=densities['slag'])
densities['flyash'] = st.sidebar.number_input('Flyash density (kg/m³)', value=densities['flyash'])
densities['coarseaggregate'] = st.sidebar.number_input('Coarse aggregate density (kg/m³)', value=densities['coarseaggregate'])
densities['fineaggregate'] = st.sidebar.number_input('Fine aggregate density (kg/m³)', value=densities['fineaggregate'])
densities['water'] = st.sidebar.number_input('Water density (kg/m³)', value=densities['water'])

# User input for target compressive strength
target_compressive_strength = st.number_input('Enter the target compressive strength (20 MPa - 70 MPa)', min_value=20.0, max_value=70.0, value=40.0)

# User input for carbon footprint constraint
carbon_footprint_constraint = st.number_input('Maximum allowable carbon footprint (kg CO2-eq)', value=200.0, min_value=0.0)

if st.button("Optimize Mix"):
    try:
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        # Get best parameters
        best_params = study.best_params

        # Compute results
        cement = best_params['cement']
        slag = best_params['slag']
        flyash = best_params['flyash']
        water = best_params['water']
        superplasticizer = best_params['superplasticizer']
        coarseaggregate = best_params['coarseaggregate']
        fineaggregate = best_params['fineaggregate']
        age = best_params['age']

        total_binder = cement + slag + flyash
        total_aggregate = coarseaggregate + fineaggregate
        water_to_binder_ratio = water / total_binder
        wc_ratio = water / cement
        total_volume = calculate_volume(cement, slag, flyash, water, coarseaggregate, fineaggregate)

        # Predict compressive strength using CatBoost model
        mix = np.array([[cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age]])
        predicted_compressive_strength = best_catboost.predict(mix)[0]

        total_cost = (cement * unit_costs['cement'] + slag * unit_costs['slag'] +
                      flyash * unit_costs['flyash'] + water * unit_costs['water'] +
                      superplasticizer * unit_costs['superplasticizer'] +
                      coarseaggregate * unit_costs['coarseaggregate'] +
                      fineaggregate * unit_costs['fineaggregate'])

        total_carbon_footprint = (cement * carbon_footprints['cement'] + slag * carbon_footprints['slag'] +
                                  flyash * carbon_footprints['flyash'] + water * carbon_footprints['water'] +
                                  superplasticizer * carbon_footprints['superplasticizer'] +
                                  coarseaggregate * carbon_footprints['coarseaggregate'] +
                                  fineaggregate * carbon_footprints['fineaggregate'])

        st.write("Optimized Mix:")
        st.write(f"Cement: {cement:.2f} kg")
        st.write(f"Slag: {slag:.2f} kg")
        st.write(f"Flyash: {flyash:.2f} kg")
        st.write(f"Water: {water:.2f} kg")
        st.write(f"Superplasticizer: {superplasticizer:.2f} kg")
        st.write(f"Coarse Aggregate: {coarseaggregate:.2f} kg")
        st.write(f"Fine Aggregate: {fineaggregate:.2f} kg")
        st.write(f"Age: {age:.2f} days")

        st.write(f"Predicted Compressive Strength: {predicted_compressive_strength:.2f} MPa")
        st.write(f"Total Cost: ${total_cost:.2f}")
        st.write(f"Total Carbon Footprint: {total_carbon_footprint:.2f} kg CO2-eq")

        # Calculate prediction accuracy
        target_strength = np.array([target_compressive_strength])
        prediction_accuracy = 1 - abs(predicted_compressive_strength - target_compressive_strength) / target_compressive_strength
        st.write(f"Prediction Accuracy: {prediction_accuracy:.2%}")
        
                # SHAP values
        explainer = shap.Explainer(best_catboost)
        shap_values = explainer(mix)

        # Calculate absolute mean SHAP values
        abs_shap_values = np.abs(shap_values.values).mean(axis=0)

        # Plot absolute SHAP values as bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        features = ['Cement', 'Slag', 'Fly Ash', 'Water', 'Super Plasticizer',
                    'Coarse Aggregates', 'Fine Aggregates', 'Age']
        ax.barh(features, abs_shap_values, color='steelblue')
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlabel('Mean Absolute SHAP Value', fontsize=16)
        plt.tight_layout()

        # Save plot to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, caption="Feature Importance", use_column_width=True)

        # Add footnote
        st.markdown("""
        **Notes**: 
        1. The predicted strength was intended for concrete cubic compressive strength in Megapascal.
        2. The values of the mix constituents predicted in kilogram were evaluated for 1 cubic meter of concrete.
        3. Please note that the model was trained using concrete mix data from Yeh (2008) and the carbon and costs inputs from Khodabakhshian et al (2018) and Thilakarathna et al (2020).
        4. The allowable carbon footprint threshold might be adjusted as the required when the compressive strength exceeds 60 MPa.
        5. The results from the model should be verified and certified by a competent structural engineer with expertise in concrete before use in practical applications.
        """)
        
        st.markdown("""
        **References**: 
        1. Khodabakhshian, A., De Brito, J., Ghalehnovi, M., Shamsabadi, E.A. (2018). Mechanical, environmental and economic performance of structural concrete containing silica fume and marble industry waste powder. Construction and Building Materials. 169, pp. 237–251. https://doi.org/10.1016/j.conbuildmat.2018.02.192
        2. Thilakarathna, P.S.M., Seo, S., Kristombu Baduge, K.S., Lee, H., Mendis, P., Foliente, G. (2020). Embodied carbon analysis and benchmarking emissions of high and ultra-high strength concrete using machine learning algorithms. Journal of Cleaner Production. https://doi.org/10.1016/j.jclepro.2020.121281
        3. Yeh, I. C. (1998). Modeling of strength of high performance concrete using artificial neural networks. Cement and Concrete Research. 28 (12), pp. 1797-1808. https://doi.org/10.1016/S0008-8846(98)00165-3
        """)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
# Adding a footer with contact information
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px;
    font-size: 12px;
    color: #6c757d;
}
</style>
<div class="footer">
    <p>© 2024 My Streamlit App. All rights reserved. |Temitope E. Dada | For Queries: <a href="mailto:t.e.dada@liverpool.ac.uk">t.e.dada@liverpool.ac.uk</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)   