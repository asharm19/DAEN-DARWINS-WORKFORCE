import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from scipy.optimize import linprog
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import plotly.express as px
from scipy.optimize import linprog
# --- Preprocessing Function ---
def preprocess_hiring_data(df):
    df = df.dropna(subset=["lat", "lng", "Cost of Living Index", 
                           "Avg.OfficeRent_per_sqft", "Annual Commute Cost", 
                           "Salary_in_office"])

    df['geometry'] = df.apply(lambda row: Point(row['lng'], row['lat']), axis=1)

    if df['Experience'].dtype == 'O':
        exp_map = {"Entry": 1, "Mid": 2, "Senior": 3}
        df["Experience_Level"] = df["Experience"].map(exp_map).fillna(2)
    else:
        df["Experience_Level"] = df["Experience"].fillna(df["Experience"].median())

    df["Rent_Commute"] = df["Avg.OfficeRent_per_sqft"] * df["Annual Commute Cost"]
    df["Experience_Cost"] = df["Experience_Level"] * df["Cost of Living Index"]

    df["log_commute"] = np.log1p(df["Annual Commute Cost"])
    df["log_rent"] = np.log1p(df["Avg.OfficeRent_per_sqft"])

    feature_cols = [
        "Cost of Living Index", 
        "Avg.OfficeRent_per_sqft", 
        "Annual Commute Cost", 
        "Experience_Level", 
        "Rent_Commute", 
        "Experience_Cost",
        "log_commute",
        "log_rent"
    ]

    return df, feature_cols

# --- Load Data ---
df = pd.read_csv("/Users/mogana/myProjects/DAEN-DARWINS-WORKFORCE/Notebook code files/Data/Final_Dataset.csv")
df, features = preprocess_hiring_data(df)
geo_df = gpd.GeoDataFrame(df, geometry='geometry')

# --- Streamlit UI ---
st.set_page_config(page_title="Hiring Optimization Dashboard", layout="wide")
st.title("💼 Hiring Cost Optimization Portal")

option = st.sidebar.radio("Choose View:", ["Hiring Cost Map Visualization", "Interactive Optimization Tool"])

if option == "Hiring Cost Map Visualization":
    st.header("Tableau Dashboard")
    st.markdown("[Click here to view the Tableau Dashboard](https://public.tableau.com/views/clustering_dashboard/Dashboard?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)", unsafe_allow_html=True)

elif option == "Interactive Optimization Tool":
    # --- User Inputs ---
    sector = st.selectbox("Select Sector", df['Sector'].unique())
    filtered_roles = df[df['Sector'] == sector]['Role'].unique()
    role = st.selectbox("Select Role", filtered_roles)
    experience = st.selectbox("Select Experience Level", ["Entry", "Mid", "Senior"])
    budget = st.number_input("Enter Your Budget ($)", min_value=10000, value=80000)
    hire_type = st.selectbox("Hiring Type", ["Remote", "In-Office", "Both"])

    st.subheader("📍 Compare Hiring Costs Across States")


    # States to compare
    selected_states = st.multiselect("Select States to Compare", df['State'].unique())

    if selected_states:
        # Filter data by selected inputs
        filtered_df = df[
            (df['Sector'] == sector) &
            (df['Role'] == role) &
            (df['Experience'] == experience) &
            (df['State'].isin(selected_states))
        ].copy()

        # Determine which cost to focus on
        if hire_type == "Remote":
            filtered_df["Selected Cost"] = filtered_df["Total Remote salary"]
            filtered_df["Within Budget"] = filtered_df["Selected Cost"] <= budget
            st.dataframe(filtered_df[[
                "City", "State", "Total Remote salary", "Within Budget"
            ]])

        elif hire_type == "In-Office":
            filtered_df["Selected Cost"] = filtered_df["Total In-office salary"]
            filtered_df["Within Budget"] = filtered_df["Selected Cost"] <= budget
            st.dataframe(filtered_df[[
                "City", "State", "Total In-office salary", "Within Budget"
            ]])

        else:  # Both
            filtered_df["Remote Within Budget"] = filtered_df["Total Remote salary"] <= budget
            filtered_df["Office Within Budget"] = filtered_df["Total In-office salary"] <= budget

            filtered_df["Most Cost Effective"] = np.where(
                filtered_df["Total Remote salary"] < filtered_df["Total In-office salary"],
                "Remote",
                "In-Office"
            )

            st.dataframe(filtered_df[[
                "City", "State",
                "Total In-office salary", "Total Remote salary",
                "Most Cost Effective",
                "Remote Within Budget", "Office Within Budget"
            ]])

            # Summary Table
            state_summary = filtered_df.groupby("State").agg({
                "Total In-office salary": "mean",
                "Total Remote salary": "mean"
            }).reset_index()

            state_summary["Most Cost Effective"] = np.where(
                state_summary["Total Remote salary"] < state_summary["Total In-office salary"],
                "Remote",
                "In-Office"
            )

            st.markdown("### 🧮 Summary by State")
            st.dataframe(state_summary)

            # Chart
            fig = px.bar(
                state_summary.melt(id_vars=["State", "Most Cost Effective"],
                                value_vars=["Total In-office salary", "Total Remote salary"]),
                x="State",
                y="value",
                color="variable",
                barmode="group",
                title="State-wise Average In-Office vs Remote Salary"
            )
            st.plotly_chart(fig)


    # --- Filter Data ---
    exp_map = {"Entry": 1, "Mid": 2, "Senior": 3}
    experience_level = exp_map[experience]
    role_df = geo_df[(geo_df['Role'] == role) & (geo_df['Experience_Level'] == experience_level)].copy()

    # --- Feature Scaling & Prediction ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(role_df[features])
    y = role_df["Salary_in_office"].values.reshape(-1, 1)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_scaled, y.ravel())
    role_df["RF_Predicted_Salary"] = rf.predict(X_scaled)

    # --- Optimization Model ---
    role_df['Cost Difference'] = role_df['Total In-office salary'] - role_df['Total Remote salary']
    role_df = role_df.sort_values(by="Cost Difference")

    # --- Filter within Budget ---
    if hire_type == "Remote":
        filtered_df = role_df[role_df['Total Remote salary'] <= budget]
    elif hire_type == "In-Office":
        filtered_df = role_df[role_df['Total In-office salary'] <= budget]
    else:
        filtered_df = role_df[(role_df['Total Remote salary'] <= budget) | (role_df['Total In-office salary'] <= budget)]

    # --- Top Cost-effective Cities ---
    st.subheader("Top Cost-Effective Cities")
    if hire_type in ["Remote", "Both"]:
        st.write("**Remote:**")
        st.dataframe(filtered_df.nsmallest(5, 'Total Remote salary')[['City', 'State', 'Total Remote salary']])

    if hire_type in ["In-Office", "Both"]:
        st.write("**In-Office:**")
        st.dataframe(filtered_df.nsmallest(5, 'Total In-office salary')[['City', 'State', 'Total In-office salary']])

    # --- Detailed Cost Breakdown ---
    st.subheader("Detailed Cost Comparison")
    comparison_cols = ['City', 'State', 'Salary_in_office', 'Office Rent', 'Annual Commute Cost', 
                    'Remote Work Expenses', 'Total In-office salary', 'Total Remote salary', 'Cost Difference']
    st.dataframe(filtered_df[comparison_cols].sort_values(by='Total In-office salary').reset_index(drop=True))

    # --- Verdict Recommendation ---
    st.subheader("🏆 Recommendation")
    best_remote = filtered_df.nsmallest(1, 'Total Remote salary').iloc[0]
    best_office = filtered_df.nsmallest(1, 'Total In-office salary').iloc[0]

    if best_remote['Total Remote salary'] < best_office['Total In-office salary']:
        percent = ((best_office['Total In-office salary'] - best_remote['Total Remote salary']) / best_office['Total In-office salary']) * 100
        st.markdown(f"Based on your budget and selected role, it's **{percent:.2f}% cheaper** to hire **remotely** in **{best_remote['City']}, {best_remote['State']}** vs in-office in **{best_office['City']}, {best_office['State']}**.")
    else:
        percent = ((best_remote['Total Remote salary'] - best_office['Total In-office salary']) / best_remote['Total Remote salary']) * 100
        st.markdown(f"Based on your budget and selected role, it's **{percent:.2f}% cheaper** to hire **in-office** in **{best_office['City']}, {best_office['State']}** vs remotely in **{best_remote['City']}, {best_remote['State']}**.")

    # --- Further Prediction Models ---
    # Handle missing values
    columns_to_impute = [
        'Salary_in_office', 'Salary_remote',
        'Office Rent', 'Annual Commute Cost',
        'In-Office Expenses', 'Remote Work Expenses'
    ]
    imputer = SimpleImputer(strategy='mean')
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

    # Recompute total salaries
    df['Total In-office salary'] = (
        df['Salary_in_office'] + df['Office Rent'] +
        df['Annual Commute Cost'] + df['In-Office Expenses']
    )
    df['Total Remote salary'] = (
        df['Salary_remote'] + df['Remote Work Expenses']
    )

    # Drop rows where target vars are still NaN (edge case)
    df.dropna(subset=['Total In-office salary', 'Total Remote salary'], inplace=True)

    # Train predictive models
    X = df[['Salary_in_office', 'Salary_remote']]
    y_office = df['Total In-office salary']
    y_remote = df['Total Remote salary']

    model_office = LinearRegression().fit(X, y_office)
    model_remote = LinearRegression().fit(X, y_remote)

    # Predict total salaries based on budget input
    pred_input = [[budget, budget]]
    predicted_office_salary = model_office.predict(pred_input)[0]
    predicted_remote_salary = model_remote.predict(pred_input)[0]

    st.subheader("Predicted Total Salaries")
    st.write(f"**In-office**: ${predicted_office_salary:,.2f}")
    st.write(f"**Remote**: ${predicted_remote_salary:,.2f}")

    # --- Optimize Hiring Strategy ---
    # def optimize_hiring_strategy(df, budget):
    #     role_df = df[(df['Total In-office salary'] <= budget) | (df['Total Remote salary'] <= budget)].copy()

    #     if role_df.empty:
    #         st.write("⚠️ No options within the given budget.")
    #         return

    #     # Create cost vectors: remote and in-office costs for each city/role
    #     c = role_df[['Total Remote salary', 'Total In-office salary']].mean().values

    #     # Set up the constraints: Ensure each city/role's cost stays within budget
    #     A_ub = [[1, 0], [0, 1]]  # Constraints for remote and in-office costs
    #     b_ub = [budget, budget]   # Budget constraint

    #     # Perform the optimization
    #     result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

    #     if result.success:
    #         st.subheader("🧮 Optimization Results")
    #         st.write("Optimal hiring strategy: ", result.x)
    #     else:
    #         st.write("Optimization failed. Try adjusting the budget.")

    # optimize_hiring_strategy(df, budget)


    # --- Scatter Plot ---
    st.subheader("📈 Salary vs Total Cost Analysis")
    plot_df = filtered_df.copy()
    plot_df['Hiring Type'] = np.where(plot_df['Total Remote salary'] < plot_df['Total In-office salary'], 'Remote', 'In-Office')
    plot_df['Predicted Salary'] = plot_df['RF_Predicted_Salary']
    plot_df['Total Cost'] = np.where(
        plot_df['Hiring Type'] == 'Remote',
        plot_df['Total Remote salary'],
        plot_df['Total In-office salary']
    )

    fig = px.scatter(
        plot_df,
        x='Predicted Salary', y='Total Cost', color='Hiring Type',
        hover_data=['City', 'State'],
        title='Predicted Salary vs Total Cost by Hiring Type'
    )
    st.plotly_chart(fig)

    # --- Map ---
    st.subheader("Hiring Locations Map")
    st.map(filtered_df.rename(columns={"lat": "latitude", "lng": "longitude"}))

    