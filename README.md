# Unlocking Customer Insights: A Statistical Investigation

**Business Problem**  
A mid-sized Indian retail company had 10,675 customer records but no clarity on spending patterns and demographics. Task was to validate assumptions and support data-driven decisions.

**My Approach**  
1. **Data Understanding**: Loaded data, checked nulls, data types - Age, Gender, Education, MonthlySpend, etc
2. **Descriptive Stats**: Mean, median, std dev for Age, MonthlySpend, DaysSinceLastInteraction
3. **Visualizations**: Histograms, Boxplots, Scatterplots Age vs Spend, Bar charts for Gender/Education/State, KDE plots
4. **Bivariate Analysis**: Correlation matrix, Crosstab Gender vs Married, Grouped stats by State/Education
5. **Hypothesis Testing**: 
   - T-test: Males vs Females spending
   - ANOVA: Education level impact on spend  
   - Chi-square: Marital status vs NumPets
   - Correlation: Age vs Engagement

**Key Insights**  
- [Result likh de baad me jab analysis kar legi] Ex: Master's degree customers spend 18% more
- [Ek aur finding] Ex: Florida/Texas show highest spending variability

**Tech Stack**  
Python | Pandas | NumPy | Matplotlib | Seaborn | Scipy

**Dataset**: 10,675 rows, 1000+ unique customers with behavioral + demographic data
