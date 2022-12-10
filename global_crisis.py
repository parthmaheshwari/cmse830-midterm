import streamlit as st
import seaborn as sns
import pandas as pd
import seaborn as sns
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
from iso3166 import countries
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


def generate_roc_plot(fpr, tpr, thresholds):
    roc_df = pd.DataFrame()
    roc_df['fpr'] = fpr
    roc_df['tpr'] = tpr
    roc_df['thresholds'] = thresholds
    roc_line = alt.Chart(roc_df).mark_line(color = 'red').encode(
                                                        alt.X('fpr', title="false positive rate"),
                                                        alt.Y('tpr', title="true positive rate"))
    roc = alt.Chart(roc_df).mark_area(fillOpacity = 0.5, fill = 'red').encode(alt.X('fpr', title="false positive rate"),
                                                            alt.Y('tpr', title="true positive rate"))
    baseline = alt.Chart(roc_df).mark_line(strokeDash=[20,5], color = 'black').encode(alt.X('thresholds', scale = alt.Scale(domain=[0, 1]), title=None),
                                                        alt.Y('thresholds', scale = alt.Scale(domain=[0, 1]), title=None))
    c = roc_line + roc + baseline.properties(title='ROC Curve').interactive()
    return c


def get_plots(mod, X_train, X_test, y_train, y_test):
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, mod.predict_proba(X_test)[:,1])
    cm = confusion_matrix(y_pred, y_test)
    c = generate_roc_plot(fpr, tpr, thresholds)
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True,fmt='g')

    precision = round(precision_score(y_test, y_pred),3)
    recall = round(recall_score(y_test, y_pred),3)
    accuracy = round(accuracy_score(y_test, y_pred),3)
    f1 = round(f1_score(y_test, y_pred),3)
    
    return c, fig, precision, recall, accuracy, f1


def f(x):
    # Function generating country codes for chloropeth map
    try:
        return countries.get(x).numeric
    except:
        return None

st.set_page_config(layout="wide")
st.write("""
    # Global Crises Data by Country
    How different macroeconomics factor can help us predict systemic crisis in different countries. 
    Shown below: Countries in Africa 
    """)

# Dataset source
st.write("Dataset source: [https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx](https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx)")
st.write("Code: [Github](https://github.com/parthmaheshwari/cmse830-midterm)")

tab1, tab2 = st.tabs(["Understanding Crises", "Predicting Crises"])

with tab1:
    # Preprocessing

    df = pd.read_csv("african_crises.csv")
    # st.set_page_config(layout="wide")
    # data = df[["exch_usd", "inflation_annual_cpi", "year", "systemic_crisis","gdp_weighted_default","country"]]
    # data = data[(data["inflation_annual_cpi"]<100)]
    # df = data.drop('systemic_crisis', axis=1)
    # df_norm = (df-df.min())/(df.max()-df.min())
    # df_norm = pd.concat((df_norm, data.systemic_crisis), 1)
    df["banking_crisis"][df["banking_crisis"]=="crisis"] = 1
    df["banking_crisis"][df["banking_crisis"]=="no_crisis"] = 0
    df["banking_crisis"] = pd.to_numeric(df["banking_crisis"])
    df = df[df["currency_crises"]<=1]
    df["year"] = pd.to_datetime(df.year, format='%Y')
    scaled_df = df.copy()

    # Country dropdown
    countries_list = list(df["country"].unique())
    st.header("Varying trends across time and countries -")
    country = st.selectbox("Select a country: ",countries_list)

    # Multiselect dropdown for types of crises
    crisis_options = st.multiselect(
        'Types of crises:',
        ['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'],
        default=['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'])

    # Multiselect dropdown for Economic parameters
    eco_ops = st.multiselect(
        'Macroeconomic parameters:',
        ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence'],
        default = ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'inflation_annual_cpi', 'independence'])

    # Scaling the binary columns for better visibility - Checkbox + text input
    agree = st.checkbox('Scaling enabled')
    sf = st.number_input('Select scaling factor', min_value=1, max_value=100, value = 1)

    if agree:
        scaled_df = df.copy()
        scaled_df["banking_crisis"]*=sf
        scaled_df["currency_crises"]*=sf
        scaled_df["inflation_crises"]*=sf
        scaled_df["systemic_crisis"]*=sf
        scaled_df["sovereign_external_debt_default"]*=sf
        scaled_df["domestic_debt_in_default"]*=sf
        scaled_df["independence"]*=sf


    ###
    # Multi-Line chart 
    ###
    # preprocessing
    country_df = scaled_df[scaled_df["country"]==country]
    country_df.set_index("year", inplace=True)
    source = country_df.drop(columns=["country","cc3","case"])
    column_list = [i for i in list(country_df) if i not in []]
    source = source[crisis_options+eco_ops]
    source = source.reset_index().melt('year', var_name='category', value_name='y')

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['year'], empty='none')

    # The basic line
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x='year:T',
        y='y:Q',
        color='category:N',
        strokeDash=alt.condition(
            (alt.datum.category == 'exch_usd') | (alt.datum.category == 'domestic_debt_in_default') | (alt.datum.category == 'sovereign_external_debt_default') | (alt.datum.category == 'gdp_weighted_default') | (alt.datum.category == 'inflation_annual_cpi') | (alt.datum.category == 'independence'),
            alt.value([5, 5]),  # dashed line: 5 pixels  dash + 5 pixels space
            alt.value([0]),  # solid line
        )
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x='year:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'y:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='year:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    c = alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=600, height=300
    ).interactive()

    st.altair_chart(c, use_container_width=True)
    #----------------------------------------------------------------------

    ###
    # Stacked bar chart 
    ###

    st.header("Which countries are more vulnerable to crises")

    # Time Slider
    values = st.slider(
        'Select a range of years',
        1870, 2013, (1870, 2013))

    # Slicing datat as per selected time
    selector = alt.selection_single(encodings=['x', 'color'])
    df_y = df[(df["year"]>=f"01-01-{values[0]}")&(df["year"]<=f"01-01-{values[-1]}")]

    # Plot with selection, gray color used for showing focus on selected part
    c2 = alt.Chart(df_y).transform_fold(
    ['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis'],
    as_=['column', 'value']
    ).mark_bar().encode(
    x=alt.X('country:N', title="Countries"),
    y=alt.Y('sum(value):Q', title="Total no. of crises"),
    color=alt.condition(selector, 'column:N', alt.value('lightgray'))
    ).add_selection(
        selector
    ).interactive()

    st.altair_chart(c2, use_container_width=True)
    #----------------------------------------------------------------------


    ###
    # Concatenated scatter plots for finding cause of crises using hues
    ###

    st.header("What causes a crisis?")

    # Radio button
    crisis = st.radio(
        "Select the type of crises to analyse (use this radio button for chloropeth map as well(last))",
        ('inflation_crises', 'systemic_crisis', 'banking_crisis','currency_crises'))

    # 
    c3 = alt.Chart(country_df.reset_index()).mark_circle(size=60).encode(
        x = 'inflation_annual_cpi:Q',
        y = 'exch_usd:Q',
        color=f'{crisis}:N',
        tooltip=['sovereign_external_debt_default:N', 'domestic_debt_in_default:N','year:T']
    ).properties(
        width=300,
        height=300
    ).interactive()


    c4 = alt.Chart(country_df.reset_index()).mark_circle(size=60).encode(
        x = 'inflation_annual_cpi:Q',
        y = 'gdp_weighted_default:Q',
        color=f'{crisis}:N',
        tooltip=['sovereign_external_debt_default:N', 'domestic_debt_in_default:N','year:T']
    ).properties(
        width=300,
        height=300
    ).interactive()

    hc = alt.hconcat(c3, c4)
    st.altair_chart(hc, use_container_width=True)
    #----------------------------------------------------------------------

    ###
    # Dynamic Pie chart
    ###

    st.header("Does defaulting on a debt cause a crisis?")

    # Slider for slicing data based on different scenarios
    debt = st.select_slider(
        'Select the type of debt in DEFAULT:',
        options=['No Debt', 'Domestic Debt', 'International Debt', 'International + Domestic Debt'])

    # Adding a new column relevant to the story of the plot
    df["no_crises"] = (df["inflation_crises"]==0)&(df["currency_crises"]==0)&(df["systemic_crisis"]==0)&(df["banking_crisis"]==0)
    df["no_crises"] = df["no_crises"].astype(int)

    # Slicing dataframe based on selection
    if debt == "No Debt":
        crisis_df = df[(df["domestic_debt_in_default"]==0)&(df["sovereign_external_debt_default"]==0)][['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis', 'no_crises','year']]
    elif debt == "Domestic Debt":
        crisis_df = df[(df["domestic_debt_in_default"]==1)&(df["sovereign_external_debt_default"]==0)][['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis', 'no_crises','year']]
    elif debt == "International Debt":
        crisis_df = df[(df["domestic_debt_in_default"]==0)&(df["sovereign_external_debt_default"]==1)][['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis', 'no_crises','year']]
    else:
        crisis_df = df[(df["domestic_debt_in_default"]==1)&(df["sovereign_external_debt_default"]==1)][['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis', 'no_crises','year']]
        
    # Aggregation for Pie chart
    crisis_df = crisis_df.reset_index().melt('year', var_name='category', value_name='y')
    count_df = pd.DataFrame(crisis_df[crisis_df["y"]==1]["category"].value_counts()).reset_index()

    # Pie chart plot
    c5 = alt.Chart(count_df).mark_arc().encode(
        theta=alt.Theta(field="category", type="quantitative"),
        color=alt.Color(field="index", type="nominal"),
    ).interactive()

    st.altair_chart(c5, use_container_width=True)
    #----------------------------------------------------------------------

    ###
    # Catplot with gaussian jitter
    ###

    st.header("Can one crisis cause another?")

    # Preprocessing for time series analysis of crises
    crisis_df1 = df[df["country"]==country][['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis','year']]
    crisis_df1 = crisis_df1.melt('year', var_name='category', value_name='y')
    crisis_df1 = crisis_df1[crisis_df1["y"]==1]

    # Catplot
    c6 =  alt.Chart(crisis_df1, width=40).mark_circle(size=60).encode(
    x=alt.X(
        'jitter:Q',
        title=None,
        axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
        scale=alt.Scale(),
    ),
    y=alt.Y('year:T'),
    color=alt.Color('category:N', legend=None),
    column=alt.Column(
        'category:N',
        header=alt.Header(
            labelAngle=-90,
            titleOrient='top',
            labelOrient='bottom',
            labelAlign='right',
            labelPadding=3,
        ),
    ),
    ).transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    ).properties(
        width=150,
        height=400
    ).interactive()

    st.altair_chart(c6)
    #----------------------------------------------------------------------

    ###
    # Chrolopeth map
    ###

    st.header("The big picture!")

    year1 = st.slider('Select the year - ', 1870, 2013, 2000)
    worldmap_url = "https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/world-110m.json"
    df["country_code"] = df["country"].apply(f)
    countries1 = alt.topo_feature(worldmap_url,"countries")
    c7 = alt.Chart(countries1).mark_geoshape().encode(
        color=f'{crisis}:N'
    ).transform_lookup(
        lookup='id',
        from_=alt.LookupData(df[df["year"]==f"{year1}-01-01"], 'country_code', [crisis])
    ).project(
        type='naturalEarth1'
    ).properties(
        width=500,
        height=300,
        title='Crises over years in Africa'
    ).interactive()
    st.altair_chart(c7, use_container_width=True)
    #----------------------------------------------------------------------

    # Additional(useless/non-functional) plots
    #1. 
    # c = alt.Chart(df[df["country"]==country]).mark_line().encode(
    #     x = 'year',
    #     y='inflation_annual_cpi'
    #     ).interactive()

    # st.altair_chart(c, use_container_width=True)

    #3. 
    # fig = px.scatter_3d(data, x = 'exch_usd', 
    #                     y = 'inflation_annual_cpi', 
    #                     z = 'year',
    #                     color = 'country',
    #                     size_max = 20, 
    #                     opacity = 0.5)

    # st.plotly_chart(fig, use_container_width=True)



with tab2:
    if 'knn' not in st.session_state:
	    st.session_state.knn = 0
    if 'lr' not in st.session_state:
	    st.session_state.lr = 0
    if 'svc' not in st.session_state:
	    st.session_state.svc = 0
    if 'mlp' not in st.session_state:
	    st.session_state.mlp = 0
    if 'rf' not in st.session_state:
	    st.session_state.rf = 0

    df = pd.read_csv("african_crises.csv")
    df["banking_crisis"][df["banking_crisis"]=="crisis"] = 1
    df["banking_crisis"][df["banking_crisis"]=="no_crisis"] = 0
    df["banking_crisis"] = pd.to_numeric(df["banking_crisis"])

    y_name = st.selectbox("Select the predicted variable: ",['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'])
    other_cols = [i for i in ['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'] if i != y_name]

    st.header("Feature Engineering")
    
    st.subheader("Select Input Features -")
    
     # Multiselect dropdown for Economic parameters
    col_names = st.multiselect(
        'Input features:',
        ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence']+other_cols,
        default = ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence']+other_cols)


    # Label Encoder
    agree = st.checkbox('Encode categorical variables?')
    if agree:
        col_names.append("country_cat")

    df["country_cat"] = LabelEncoder().fit_transform(df["country"])
    df.drop(columns=["cc3","case","year","country"],inplace=True)

    c11 = alt.Chart(df).mark_bar().encode(
    alt.Y(f'{y_name}:N'),
    alt.X(f'count({y_name}):Q'))
    st.altair_chart(c11, use_container_width=True)

    # Shift columns
    st.subheader("Shift columns - ")
    prev_col_names = st.multiselect(
        'Columns to be shifted:',
        ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence']+other_cols,
        default = ['exch_usd', 'gdp_weighted_default', 'inflation_annual_cpi'])

    nshifts = st.number_input('Number of shifts', value=1)

    for col in prev_col_names:
        df[f"prev{nshifts}_{col}"] = df.groupby('country_cat')[col].shift(nshifts)

    df.dropna(inplace=True)

    # Oversample
    st.subheader("Oversampling -")
    percent_os = st.slider('Set Minority class-Majority class ratio:', 0.00, 1.00, 1.00, step = 0.05)
    X = df.drop(columns = [y_name])
    y = df[y_name]
    X_resampled, y = SMOTE(sampling_strategy=percent_os, random_state =42).fit_resample(X, y)
    X = pd.DataFrame(X_resampled, columns=X_resampled.columns)

    c12 = alt.Chart(pd.DataFrame(y)).mark_bar().encode(
    alt.Y(f'{y_name}:N'),
    alt.X(f'count({y_name}):Q'))
    st.altair_chart(c12, use_container_width=True)

    
    st.subheader("PCA -")

    st.header("Train Test Split")
    # Train Test split
    percent_test = st.slider('Set Train-Test ratio:', 0.00, 1.00, 0.30, step = 0.05)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=percent_test,random_state=42)

    st.subheader("Scaling -")

    scaler_type = st.radio(
        "Select the type of scaling operation/normalization -",
        ('No scaling', 'MinMax', 'Standard', 'Normalize'))

    if scaler_type == "MinMax":
        scaler = MinMaxScaler()
    elif scaler_type == "Standard":
        scaler = StandardScaler()
    elif scaler_type == "Normalize":
        scaler = Normalizer()

    if scaler_type!="No scaling":
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    st.header("Train, Evaluate and Tune Estimators with Cross Validation")
    df.dropna(inplace=True)

    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

    with col1:
        st.subheader("Logistic Regression")
        parameters_lr = {"penalty":('l1', 'l2', 'elasticnet'), "solver":('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), "C":[1,0.1,0.01,0.001]}
        penalty = st.selectbox('Select penalty -',parameters_lr["penalty"],index=1)
        solver = st.selectbox("Select solver -", parameters_lr["solver"],index=1)
        c = st.select_slider('Select regularization strength -', parameters_lr["C"],value=0.1)

        lr = LogisticRegression(penalty = penalty, solver = solver, C = c)
        c13, fig_lr, precision_lr, recall_lr, f1_lr, accuracy_lr = get_plots(lr, X_train, X_test, y_train, y_test)

        
        if st.button('Evaluate LR'):
            st.session_state.lr += 1

        if st.session_state.lr > 0:       
            st.altair_chart(c13, use_container_width=True)
            st.pyplot(fig_lr)

            if 'lr_max_precision' not in st.session_state:
	            st.session_state.lr_max_precision, st.session_state.lr_max_recall, st.session_state.lr_max_f1, st.session_state.lr_max_accuracy= precision_lr, recall_lr, f1_lr, accuracy_lr                

            st.metric(label="Precision", value=precision_lr, delta = str(round(precision_lr-st.session_state.lr_max_precision,3)))
            st.metric(label="Recall", value=recall_lr, delta = str(round(recall_lr-st.session_state.lr_max_recall,3)))
            st.metric(label="F1-score", value=f1_lr, delta = str(round(f1_lr-st.session_state.lr_max_f1,3)))
            st.metric(label="Accuracy", value=accuracy_lr, delta = str(round(accuracy_lr-st.session_state.lr_max_accuracy,3)))
            st.session_state.lr_max_precision = max(st.session_state.lr_max_precision, precision_lr)
            st.session_state.lr_max_recall = max(st.session_state.lr_max_recall, recall_lr)
            st.session_state.lr_max_f1 = max(st.session_state.lr_max_f1, f1_lr)
            st.session_state.lr_max_accuracy = max(st.session_state.lr_max_accuracy, accuracy_lr)

        if st.button('Tune LR with CV'):
            clf = GridSearchCV(lr, parameters_lr, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)


    with col2:
        st.subheader("Support Vector Machine")
        parameters_svc = {"kernel":('linear', 'poly', 'rbf', 'sigmoid'), "gamma":('scale', 'auto'), "C":[1,0.1,0.01,0.001]}
        kernel = st.selectbox('Select kernel -',parameters_svc["kernel"],index=2)
        gamma = st.selectbox("Select gamma -", parameters_svc["gamma"],index=0)
        c = st.select_slider('Select regularization parameter -', options=parameters_svc["C"],value=1)

        svc = SVC(kernel = kernel, gamma = gamma, C = c, probability=True, max_iter=10000)
        c14, fig_svc, precision_svc, recall_svc, f1_svc, accuracy_svc = get_plots(svc, X_train, X_test, y_train, y_test)
        
        if st.button('Evaluate SVC'):
            st.session_state.svc += 1

        if st.session_state.svc > 0:       
            st.altair_chart(c14, use_container_width=True)
            st.pyplot(fig_svc)
            st.metric(label="Precision", value=precision_svc)
            st.metric(label="Recall", value=recall_svc)
            st.metric(label="F1-score", value=f1_svc)
            st.metric(label="Accuracy", value=accuracy_svc)

        if st.button('Tune SVC with CV(Dont)'):
            clf = GridSearchCV(svc, parameters_svc, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)


    with col3:
        st.subheader("K Neighbors Classifier")
        parameters_knn = {"weights":('uniform', 'distance'), "algorithm":('auto', 'ball_tree', 'kd_tree', 'brute'), "n_neighbors":list(range(1,30,2))}
        weights = st.selectbox('Select weight function -',parameters_knn["weights"],index=0)
        algorithm = st.selectbox("Select algorithm -", parameters_knn["algorithm"],index=0)
        n_neighbors = st.select_slider('Select number of neighbors -', options=parameters_knn["n_neighbors"],value=5)
        
        knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights=weights, algorithm=algorithm)
        c15, fig_knn, precision_knn, recall_knn, f1_knn, accuracy_knn = get_plots(knn, X_train, X_test, y_train, y_test)

        if st.button('Evaluate KNN'):
            st.session_state.knn += 1

        if st.session_state.knn > 0:
            st.altair_chart(c15, use_container_width=True)
            st.pyplot(fig_knn)
            if 'knn_max_precision' not in st.session_state:
	            st.session_state.knn_max_precision, st.session_state.knn_max_recall, st.session_state.knn_max_f1, st.session_state.knn_max_accuracy= precision_knn, recall_knn, f1_knn, accuracy_knn                

            st.metric(label="Precision", value=precision_knn, delta = str(round(precision_knn-st.session_state.knn_max_precision,3)))
            st.metric(label="Recall", value=recall_knn, delta = str(round(recall_knn-st.session_state.knn_max_recall,3)))
            st.metric(label="F1-score", value=f1_knn, delta = str(round(f1_knn-st.session_state.knn_max_f1,3)))
            st.metric(label="Accuracy", value=accuracy_knn, delta = str(round(accuracy_knn-st.session_state.knn_max_accuracy,3)))
            st.session_state.knn_max_precision = max(st.session_state.knn_max_precision, precision_knn)
            st.session_state.knn_max_recall = max(st.session_state.knn_max_recall, recall_knn)
            st.session_state.knn_max_f1 = max(st.session_state.knn_max_f1, f1_knn)
            st.session_state.knn_max_accuracy = max(st.session_state.knn_max_accuracy, accuracy_knn)

        if st.button('Tune KNN with CV'):
            clf = GridSearchCV(knn, parameters_knn, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)


    with col4:
        st.subheader("Multilayer Perceptron")
        parameters_mlp = {"activation":['identity', 'logistic', 'tanh', 'relu'], "solver":('lbfgs', 'sgd', 'adam'), "hidden_layer_sizes":[(100,),(50,50,),(100,100,)]}
        activation = st.selectbox('Select activation function -', options=parameters_mlp["activation"],index=3)
        solver = st.selectbox('Select solver -',parameters_mlp["solver"], index=2)
        hidden_layer_sizes = st.text_input("Select hidden layer sizes -", "100,")
        
        mlp = MLPClassifier(activation = activation, solver=solver, hidden_layer_sizes=literal_eval(hidden_layer_sizes))
        c16, fig_mlp, precision_mlp, recall_mlp, f1_mlp, accuracy_mlp  = get_plots(mlp, X_train, X_test, y_train, y_test)

        if st.button('Evaluate MLP'):
            st.session_state.mlp += 1

        if st.session_state.mlp > 0:
            st.altair_chart(c16, use_container_width=True)
            st.pyplot(fig_mlp)
            st.metric(label="Precision", value=precision_mlp)
            st.metric(label="Recall", value=recall_mlp)
            st.metric(label="F1-score", value=f1_mlp)
            st.metric(label="Accuracy", value=accuracy_mlp)

        if st.button('Tune MLP with CV'):
            clf = GridSearchCV(mlp, parameters_mlp, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)


    with col5:
        st.subheader("Random Forest")
        parameters_rf = {"criterion":['gini', 'entropy', 'log_loss'], "n_estimators":list(range(100,501,100)), "max_depth": list(range(1,11,3))}
        criterion = st.selectbox('Select criterion for split -', options=parameters_rf["criterion"],index=0)
        n_estimators = st.select_slider('Select number of estimators -', options=parameters_rf["n_estimators"],value=100)
        max_depth = st.select_slider('Select maximum depth -', options=parameters_rf["max_depth"],value=10)
        
        rf = RandomForestClassifier(criterion = criterion, n_estimators=n_estimators, max_depth=max_depth)
        c17, fig_rf, precision_rf, recall_rf, f1_rf, accuracy_rf = get_plots(rf, X_train, X_test, y_train, y_test)

        if st.button('Evaluate RF'):
            st.session_state.rf += 1

        if st.session_state.rf > 0:
            st.altair_chart(c17, use_container_width=True)
            st.pyplot(fig_rf)
            st.metric(label="Precision", value=precision_rf)
            st.metric(label="Recall", value=recall_rf)
            st.metric(label="F1-score", value=f1_rf)
            st.metric(label="Accuracy", value=accuracy_rf)

        if st.button('Tune RF with CV'):
            clf = GridSearchCV(rf, parameters_rf, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)

    