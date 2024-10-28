import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText

# Function to send personalized emails based on cluster
def send_email(sender_email, app_password, customer_email, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = customer_email

    # Connect to Gmail's SMTP server
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.sendmail(sender_email, customer_email, msg.as_string())

# Define email templates for different clusters
EMAIL_TEMPLATES = {
    0: ("Welcome to Cluster 0", "Explore premium offers tailored just for you!"),
    1: ("Special Offers for Cluster 1", "Enjoy exclusive discounts on your purchases."),
    2: ("Hello from Cluster 2", "Check out new offers matching your preferences."),
    3: ("Exclusive Deals for Cluster 3", "Don't miss out on these limited-time deals!"),
    4: ("Welcome Back from Cluster 4", "Take advantage of exciting offers waiting for you!"),
    5: ("You have been assigned to Cluster 5", "This month end dont miss exclusive deal for you!"),
    6: ("Welcome Back from Cluster 6", "Get heavy discount on all Electronics this diwali!"),
    7: ("Welcome to Cluster 7", "Having trouble finding good deals Visit our site!"),
    8: ("You are assigned to Cluster 8", "Take advantage of exciting offers waiting for you!"),
    9: ("Welcome Back from Cluster 9", "Explore premium offers just tailored for you")
    
}

# Streamlit app setup
st.set_page_config(page_title="Customer Segmentation")
st.title("Customer Segmentation App")

# User inputs for Gmail credentials
sender_email = st.text_input("Enter your Gmail ID")
app_password = st.text_input("Enter your Google App Password", type="password")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# User input for selecting number of clusters
num_clusters = st.selectbox("Select number of clusters", [2, 3, 4, 5, 6, 7, 8, 9])

if uploaded_file is not None and sender_email and app_password:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the uploaded file
    st.write("Preview of the uploaded data:")
    st.write(df.head())
    
    # Ensure the file has all required columns
    required_columns = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 
                        'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 
                        'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 
                        'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 
                        'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 
                        'TENURE', 'email']
    
    if all(col in df.columns for col in required_columns):
        # Remove non-numeric columns such as 'email' or 'CUST_ID' from features
        non_numeric_columns = ['email', 'CUST_ID'] if 'CUST_ID' in df.columns else ['email']
        features = df.drop(non_numeric_columns, axis=1)

        # Handle missing values using an imputer (fill with mean values)
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)

        # Apply KMeans clustering with user-selected number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(features_imputed)

        # Display cluster results
        st.write("Customer clusters:")
        st.write(df[['email', 'Cluster']])

        # Plot the clusters using PCA
        pca = PCA(n_components=2)  # Reduce the data to 2D for visualization
        pca_components = pca.fit_transform(features_imputed)

        # Create a scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Cluster'], cmap='viridis')
        plt.title(f'Customer Segments (PCA Projection, {num_clusters} Clusters)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(label='Cluster')

        # Show the plot in Streamlit
        st.pyplot(plt)

        # Send emails to customers based on their clusters
        send_emails = st.button("Send Emails to Customers")
        if send_emails:
            for index, row in df.iterrows():
                customer_email = row['email']
                customer_cluster = row['Cluster']

                # Get email template for the cluster
                subject, body = EMAIL_TEMPLATES.get(
                    customer_cluster, 
                    ("Hello!", "Stay tuned for exciting updates!")  # Default template
                )

                # Send email
                send_email(sender_email, app_password, customer_email, subject, body)

            st.success("Emails sent successfully!")
    else:
        st.error("The uploaded file is missing some required columns.")
