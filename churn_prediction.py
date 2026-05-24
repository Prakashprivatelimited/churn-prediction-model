# ============================================
# CHURN PREDICTION MODEL - Starter Example
# For: Freelance Data Science Consulting
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# STEP 1: CREATE SAMPLE CUSTOMER DATA
# (In real life, the startup gives you this)
# ─────────────────────────────────────────
np.random.seed(42)
n_customers = 1000

data = pd.DataFrame({
    'customer_id':       range(1, n_customers + 1),
    'age':               np.random.randint(18, 65, n_customers),
    'months_active':     np.random.randint(1, 36, n_customers),
    'monthly_spend':     np.random.uniform(10, 500, n_customers).round(2),
    'login_frequency':   np.random.randint(0, 30, n_customers),   # logins per month
    'support_tickets':   np.random.randint(0, 10, n_customers),   # complaints
    'last_login_days':   np.random.randint(0, 90, n_customers),   # days since last login
    'plan_type':         np.random.choice(['basic', 'pro', 'enterprise'], n_customers),
})

# Simulate churn: customers who don't login & complain a lot tend to churn
churn_prob = (
    (data['last_login_days'] > 30).astype(int) * 0.4 +
    (data['support_tickets'] > 5).astype(int) * 0.3 +
    (data['monthly_spend'] < 50).astype(int) * 0.2 +
    (data['login_frequency'] < 3).astype(int) * 0.1
)
data['churned'] = (np.random.rand(n_customers) < churn_prob / churn_prob.max() * 0.6).astype(int)

print("✅ STEP 1: Data Loaded")
print(f"   Total customers: {len(data)}")
print(f"   Churned: {data['churned'].sum()} ({data['churned'].mean()*100:.1f}%)")
print(f"   Active:  {(data['churned']==0).sum()}\n")


# ─────────────────────────────────────────
# STEP 2: CLEAN & PREPARE DATA
# ─────────────────────────────────────────

# Encode categorical column (plan_type)
data['plan_basic']      = (data['plan_type'] == 'basic').astype(int)
data['plan_pro']        = (data['plan_type'] == 'pro').astype(int)
data['plan_enterprise'] = (data['plan_type'] == 'enterprise').astype(int)

# Select features
features = [
    'age', 'months_active', 'monthly_spend',
    'login_frequency', 'support_tickets', 'last_login_days',
    'plan_basic', 'plan_pro', 'plan_enterprise'
]

X = data[features]
y = data['churned']

print("✅ STEP 2: Data Cleaned & Prepared")
print(f"   Features used: {features}\n")


# ─────────────────────────────────────────
# STEP 3: TRAIN THE MODEL
# ─────────────────────────────────────────

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ STEP 3: Model Trained\n")


# ─────────────────────────────────────────
# STEP 4: TEST THE MODEL
# ─────────────────────────────────────────

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("✅ STEP 4: Model Results")
print(f"   Accuracy: {accuracy*100:.1f}%")
print()
print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))


# ─────────────────────────────────────────
# STEP 5: DELIVER RESULTS TO STARTUP
# ─────────────────────────────────────────

# Predict churn probability for ALL customers
data['churn_probability'] = model.predict_proba(
    scaler.transform(data[features])
)[:, 1]

# Classify risk level
data['risk_level'] = pd.cut(
    data['churn_probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Show top 10 at-risk customers
high_risk = data[data['risk_level'] == 'High Risk'].sort_values(
    'churn_probability', ascending=False
)

print("✅ STEP 5: High-Risk Customers (Top 10)")
print("   → Recommend targeting these customers with discounts or outreach\n")
print(high_risk[['customer_id', 'months_active', 'login_frequency',
                  'last_login_days', 'churn_probability', 'risk_level']].head(10).to_string(index=False))

print()
print("=" * 55)
print("📊 SUMMARY REPORT FOR STARTUP CLIENT")
print("=" * 55)
print(f"  Total customers analyzed : {len(data)}")
print(f"  High risk (may churn)    : {(data['risk_level']=='High Risk').sum()}")
print(f"  Medium risk              : {(data['risk_level']=='Medium Risk').sum()}")
print(f"  Low risk (safe)          : {(data['risk_level']=='Low Risk').sum()}")
print(f"  Model accuracy           : {accuracy*100:.1f}%")
print("=" * 55)
print("\n💡 RECOMMENDATION:")
print("  Focus retention efforts on High Risk customers.")
print("  Send them a discount, personal email, or a support call.")
