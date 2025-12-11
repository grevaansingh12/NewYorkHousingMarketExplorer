"""
Name: Grevaan Singh
CS230: Section 2
Data: New York Housing Market Dataset (NY-House-Dataset.csv)
URL: (To be posted on Streamlit Cloud)

Description:
This program creates an interactive web-based data explorer for the New York housing market.
It allows users to analyze property listings through customizable queries including:
- Average prices by property type across different localities
- Property searches filtered by location and price
- Price per square foot analysis by bedroom count and locality

References:
- Streamlit documentation: https://docs.streamlit.io/
- PyDeck documentation: https://deckgl.readthedocs.io/
- Pandas documentation: https://pandas.pydata.org/docs/
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk

#Page configuration
st.set_page_config(page_title="NY Housing Market Explorer", layout="wide")

#==================================================================================
#PYTHON FEATURE 1: [FUNC2P] - Function with two parameters, one with default value
#COMPLEXITY: Uses default parameter and performs multiple data cleaning operations
#==================================================================================
def loadData(filepath, dropNa=True):
    """
    Load the housing dataset and perform basic cleaning.

    Parameters:
    - filepath: path to the CSV file
    - dropNa: whether to drop rows with missing critical values (default True)

    Returns:
    - Cleaned DataFrame
    """
    dataFrame = pd.read_csv(filepath)

    #Clean price data
    if 'PRICE' in dataFrame.columns:
        dataFrame['PRICE'] = pd.to_numeric(dataFrame['PRICE'], errors='coerce')

    #Clean numeric columns
    numericCols = ['BEDS', 'BATH', 'PROPERTYSQFT', 'LATITUDE', 'LONGITUDE']
    for col in numericCols:
        if col in dataFrame.columns:
            dataFrame[col] = pd.to_numeric(dataFrame[col], errors='coerce')

    if dropNa:
        dataFrame = dataFrame.dropna(subset=['PRICE', 'LATITUDE', 'LONGITUDE'])

    #Calculate price per sqft
    dataFrame['PRICE_PER_SQFT'] = None
    validSqft = (dataFrame['PROPERTYSQFT'].notna()) & (dataFrame['PROPERTYSQFT'] > 0)
    dataFrame.loc[validSqft, 'PRICE_PER_SQFT'] = (dataFrame.loc[validSqft, 'PRICE'] /
                                                  dataFrame.loc[validSqft, 'PROPERTYSQFT'])

    return dataFrame

#=============================================================================
#PYTHON FEATURE 2: [FUNCRETURN2] - Function that returns two or more values
#=============================================================================
def getPriceStatistics(dataFrame):
    """
    Calculate price statistics for the dataset.

    Returns:
    - meanPrice: average price
    - medianPrice: median price
    - count: number of properties
    """
    meanPrice = dataFrame['PRICE'].mean()
    medianPrice = dataFrame['PRICE'].median()
    count = len(dataFrame)

    return meanPrice, medianPrice, count

#=============================================================================
#PYTHON FEATURE 3: [FUNCCALL2] - This function is called in multiple places
#First call is in the sidebar, second call is in Question 1
#=============================================================================
def formatCurrency(value):
    """Format number as currency."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"

#Load data with caching
@st.cache_data
def getData():
    return loadData("NY-House-Dataset.csv")

try:
    housingData = getData()
except:
    st.error("Error: Cannot find 'NY-House-Dataset.csv'. Please ensure the file is in the same directory.")
    st.stop()

#=====================================================================================
#STREAMLIT FEATURE 3: [ST3] - Sidebar for navigation with styling
#COMPLEXITY: Multi-page navigation system with radio buttons and dynamic info display
#=====================================================================================
st.sidebar.title("🏠 NY Housing Explorer")
st.sidebar.markdown("---")

#Navigation
currentPage = st.sidebar.radio(
    "Select a Question:",
    ["Question 1: Price by Type & Location",
     "Question 2: Property Search",
     "Question 3: Price per Sqft",
     "Interactive Map"]
)

st.sidebar.markdown("---")

#[FUNCCALL2] - First call to formatCurrency (in sidebar)
st.sidebar.text(
    f"Dataset Info\n\n"
    f"Total Properties: {len(housingData):,}\n"
    f"Price Range: ${housingData['PRICE'].min():,.0f} - ${housingData['PRICE'].max():,.0f}"
)

#=============================================================================
#QUESTION 1: How does the average price of <Type> homes vary across different <Locality> areas?
#=============================================================================
if currentPage == "Question 1: Price by Type & Location":
    st.title("📊 Question 1: Average Price by Type & Location")
    st.markdown("**How does the average price of homes vary across different localities?**")

    #=============================================================================
    #STREAMLIT FEATURE 1: [ST1] - Dropdown for property type selection
    #=============================================================================
    propertyTypes = ['All'] + sorted([t for t in housingData['TYPE'].unique() if pd.notna(t)])
    selectedType = st.selectbox("Select Property Type:", propertyTypes)

    #Filter data by property type
    if selectedType != "All":
        filteredData = housingData[housingData['TYPE'] == selectedType]
    else:
        filteredData = housingData.copy()

    #Group by locality and calculate mean
    localityPrices = filteredData.groupby('LOCALITY')['PRICE'].agg(['mean', 'count']).reset_index()
    localityPrices.columns = ['LOCALITY', 'AVG_PRICE', 'COUNT']

    #Sort by average price
    localityPrices = localityPrices.sort_values('AVG_PRICE', ascending=False)

    #Filter localities with at least 5 properties
    localityPrices = localityPrices[localityPrices['COUNT'] >= 5]

    #Display top 10 localities
    top10 = localityPrices.head(10)

    #====================================================================================
    #VISUALIZATION 1: [CHART1] - Bar chart with customized colors and labels
    #COMPLEXITY: Horizontal bar chart with custom colors, grid styling, and value labels
    #====================================================================================
    st.markdown(f"### Top 10 Localities by Average {selectedType} Price")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10['LOCALITY'], top10['AVG_PRICE'], color='#4ECDC4')

    ax.set_xlabel('Average Price ($)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Locality', fontsize=11, fontweight='bold')
    ax.set_title(f'Top 10 Localities - Average {selectedType} Price', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    #Add price labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                formatCurrency(width),
                ha='left', va='center', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    #Show statistics
    meanPrice, medianPrice, count = getPriceStatistics(filteredData)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Properties Analyzed", f"{count:,}")
    with col2:
        #[FUNCCALL2] - Second call to formatCurrency (in Question 1)
        st.metric("Average Price", formatCurrency(meanPrice))
    with col3:
        st.metric("Median Price", formatCurrency(medianPrice))

#=============================================================================
#QUESTION 2: Show all properties in <Sublocality> with a price below <amount>
#=============================================================================
elif currentPage == "Question 2: Property Search":
    st.title("🔍 Question 2: Property Search")
    st.markdown("**Show all properties in a sublocality with a price below a certain amount**")

    col1, col2 = st.columns(2)

    with col1:
        sublocalities = ['All'] + sorted([s for s in housingData['SUBLOCALITY'].unique() if pd.notna(s)])
        selectedSublocality = st.selectbox("Select Sublocality:", sublocalities)

    with col2:
        #=============================================================================
        #STREAMLIT FEATURE 2: [ST2] - Slider for price selection
        #=============================================================================
        maxPrice = st.slider(
            "Maximum Price:",
            min_value=int(housingData['PRICE'].min()),
            max_value=int(housingData['PRICE'].max()),
            value=int(housingData['PRICE'].median()),
            step=50000,
            format="$%d"
        )

    #Filter data
    searchResults = housingData[housingData['PRICE'] <= maxPrice]

    if selectedSublocality != "All":
        searchResults = searchResults[searchResults['SUBLOCALITY'] == selectedSublocality]

    st.markdown(f"### Found {len(searchResults):,} Properties")

    if len(searchResults) > 0:
        #Show min and max properties
        cheapest = searchResults['PRICE'].min()
        mostExpensive = searchResults['PRICE'].max()

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Cheapest Property**: {formatCurrency(cheapest)}")
        with col2:
            st.info(f"**Most Expensive**: {formatCurrency(mostExpensive)}")

        #Display sample properties
        st.markdown("#### Top 20 Properties by Price")
        displayCols = ['TYPE', 'PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT', 'LOCALITY', 'SUBLOCALITY']
        displayDf = searchResults[displayCols].head(20).copy()
        displayDf['PRICE'] = displayDf['PRICE'].apply(formatCurrency)
        st.dataframe(displayDf, use_container_width=True)

        #=============================================================================
        #VISUALIZATION 2: [CHART2] - Histogram showing price distribution
        #COMPLEXITY: Customized histogram with bins, colors, and grid styling
        #=============================================================================
        st.markdown("#### Price Distribution of Results")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(searchResults['PRICE'], bins=25, color='#4ECDC4', edgecolor='black')
        ax.set_xlabel('Price ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Properties', fontsize=11, fontweight='bold')
        ax.set_title('Price Distribution', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No properties found. Try adjusting your filters.")

#=============================================================================
#QUESTION 3: What is the average price per square foot for homes with at least
#<Beds> bedrooms in <Locality>?
#=============================================================================
elif currentPage == "Question 3: Price per Sqft":
    st.title("📈 Question 3: Price per Square Foot Analysis")
    st.markdown("**What is the average price per square foot for homes with at least X bedrooms in a locality?**")

    #Filter out properties without sqft data
    sqftData = housingData[housingData['PROPERTYSQFT'].notna() & (housingData['PROPERTYSQFT'] > 0)].copy()

    col1, col2 = st.columns(2)

    with col1:
        minBedrooms = st.number_input("Minimum Number of Bedrooms:", min_value=0, max_value=10, value=2)

    with col2:
        localities = ['All'] + sorted([l for l in sqftData['LOCALITY'].unique() if pd.notna(l)])
        selectedLocality = st.selectbox("Select Locality:", localities)

    #Filter data
    filteredSqft = sqftData[sqftData['BEDS'] >= minBedrooms]

    if selectedLocality != "All":
        filteredSqft = filteredSqft[filteredSqft['LOCALITY'] == selectedLocality]

    if len(filteredSqft) > 0:
        avgPricePerSqft = filteredSqft['PRICE_PER_SQFT'].mean()

        st.metric(
            f"Average Price per Sqft ({minBedrooms}+ Beds in {selectedLocality})",
            f"${avgPricePerSqft:.2f}"
        )

        #Show breakdown by bedroom count
        st.markdown("### Breakdown by Bedroom Count")

        bedroomAnalysis = []
        for beds in sorted(filteredSqft['BEDS'].unique()):
            bedData = filteredSqft[filteredSqft['BEDS'] == beds]
            avgPpsf = bedData['PRICE_PER_SQFT'].mean()
            count = len(bedData)
            bedroomAnalysis.append({
                'Bedrooms': int(beds),
                'Avg Price/SqFt': f"${avgPpsf:.2f}",
                'Property Count': count
            })

        st.table(pd.DataFrame(bedroomAnalysis))
    else:
        st.warning("No properties found matching your criteria.")

#=============================================================================
#INTERACTIVE MAP PAGE
#=============================================================================
elif currentPage == "Interactive Map":
    st.title("🗺️ Interactive Property Map")
    st.markdown("Visualize property locations across New York")

    #=============================================================================
    #VISUALIZATION 3: [MAP] - Detailed PyDeck map with custom features
    #COMPLEXITY: Advanced PyDeck map with:
    #- Custom color gradient based on price (blue=cheap, red=expensive)
    #- Interactive tooltips showing property details
    #- Multiple filter options (property type, price range, sample size)
    #- 3D perspective with pitch setting
    #- Scatterplot layer with custom styling
    #This goes beyond basic st.map() requirement
    #=============================================================================

    #ChatGPT assistance used only for explanation/summary of this section.
    #See Section 1 of accompanying AI Usage document.

    #Map filters
    st.markdown("### Map Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        mapPropertyType = st.multiselect(
            "Property Types:",
            options=sorted([t for t in housingData['TYPE'].unique() if pd.notna(t)]),
            default=sorted([t for t in housingData['TYPE'].unique() if pd.notna(t)])[:2]
        )

    with col2:
        mapPriceRange = st.slider(
            "Price Range (millions):",
            min_value=0.0,
            max_value=float(housingData['PRICE'].max() / 1000000),
            value=(0.0, 5.0),
            step=0.5
        )

    with col3:
        sampleSize = st.slider(
            "Properties to Display:",
            min_value=100,
            max_value=min(2000, len(housingData)),
            value=500,
            step=100
        )

    #Filter map data
    mapDf = housingData[
        (housingData['TYPE'].isin(mapPropertyType)) &
        (housingData['PRICE'] >= mapPriceRange[0] * 1000000) &
        (housingData['PRICE'] <= mapPriceRange[1] * 1000000)
    ].copy()

    #Sample if too many points
    if len(mapDf) > sampleSize:
        mapDf = mapDf.sample(n=sampleSize, random_state=42)

    if len(mapDf) > 0:
        st.markdown(f"### Showing {len(mapDf):,} Properties")

        #Normalize prices for color coding
        mapDf['price_normalized'] = (mapDf['PRICE'] - mapDf['PRICE'].min()) / (mapDf['PRICE'].max() -
                                                                               mapDf['PRICE'].min())

        #Create color gradient (blue to red based on price)
        def getColor(normalizedPrice):
            #Blue (cheap) to Red (expensive)
            red = int(normalizedPrice * 255)
            blue = int((1 - normalizedPrice) * 255)
            return [red, 100, blue, 180]

        mapDf['color'] = mapDf['price_normalized'].apply(getColor)

        #Create tooltip data
        mapDf['price_text'] = mapDf['PRICE'].apply(formatCurrency)
        mapDf['sqft_text'] = mapDf['PROPERTYSQFT'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
        mapDf['beds_text'] = mapDf['BEDS'].apply(lambda x: str(int(x)) if pd.notna(x) else "N/A")
        mapDf['bath_text'] = mapDf['BATH'].apply(lambda x: str(int(x)) if pd.notna(x) else "N/A")

        #Set view
        viewState = pdk.ViewState(
            latitude=mapDf['LATITUDE'].mean(),
            longitude=mapDf['LONGITUDE'].mean(),
            zoom=10,
            pitch=45
        )

        #Create scatterplot layer
        scatterLayer = pdk.Layer(
            'ScatterplotLayer',
            data=mapDf,
            get_position='[LONGITUDE, LATITUDE]',
            get_color='color',
            get_radius=50,
            pickable=True,
            opacity=0.7,
            stroked=True,
            filled=True,
            radius_min_pixels=3,
            radius_max_pixels=50,
            line_width_min_pixels=1,
            get_line_color=[255, 255, 255]
        )

        #Tooltip
        tooltip = {
            "html": "<div style='background-color: rgba(0, 0, 0, 0.8); padding: 10px; border-radius: 5px;'>"
                    "<h4 style='margin: 0; color: #4ECDC4;'>{TYPE}</h4>"
                    "<p style='margin: 5px 0; color: white;'><b>Price:</b> {price_text}</p>"
                    "<p style='margin: 5px 0; color: white;'><b>Beds:</b> {beds_text} | <b>Bath:</b> {bath_text}</p>"
                    "<p style='margin: 5px 0; color: white;'><b>Size:</b> {sqft_text} sqft</p>"
                    "<p style='margin: 5px 0; color: white;'><b>Location:</b> {LOCALITY}</p>"
                    "</div>",
            "style": {
                "backgroundColor": "transparent",
                "color": "white"
            }
        }

        #Create deck
        deck = pdk.Deck(
            layers=[scatterLayer],
            initial_view_state=viewState,
            tooltip=tooltip,
            map_style='road'
        )

        st.pydeck_chart(deck)

        st.info("💡 **Tip**: Hover over dots to see property details. Red = expensive, Blue = affordable")

        #Show summary statistics
        st.markdown("---")
        meanPrice, medianPrice, count = getPriceStatistics(mapDf)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Price", formatCurrency(meanPrice))
        with col2:
            st.metric("Median Price", formatCurrency(medianPrice))
        with col3:
            st.metric("Properties Shown", f"{count:,}")
    else:
        st.warning("No properties match your filters. Please adjust your selections.")

#Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**CS 230 Final Project**")
st.sidebar.markdown("*Grevaan Singh*")