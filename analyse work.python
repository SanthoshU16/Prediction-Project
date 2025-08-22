import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Set global aesthetics for all plots
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.linewidth': 0.5, 'grid.color': '#D3D3D3'})
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
# --- Data Preparation ---
# Load your data
data = pd.read_excel('/home/ubuntu/Documents/prediction project/Cleaned_Survey_Responses.xlsx')
# Clean and prepare data
data['AI_Fully_Replace_Professor'] = data['AI_Fully_Replace_Professor'].map({'No': 0, 'Partially': 1, 'Yes': 2})
data['Used_AI_Learning_Tools'] = data['Used_AI_Learning_Tools'].map({'Yes': 1, 'No': 0})
data['Best_Personalized_Learning'] = data['Best_Personalized_Learning'].map({'Both': 0, 'AI': 1, 'Human Professors': 2})
# Handle AI_Replacement_Challenges: Split into binary columns
challenges = ['Lack of Emotional Intelligence', 'Ethical Concerns', 'Limited Creativity', 'Risk of Bias in AI']
for challenge in challenges:
    data[challenge] = data['AI_Replacement_Challenges'].apply(lambda x: 1 if pd.notna(x) and challenge in x else 0)
data = data.drop('AI_Replacement_Challenges', axis=1)
# Drop rows with NaN in target or features
data = data.dropna(subset=['AI_Fully_Replace_Professor'] + features)
data = data[data['AI_Fully_Replace_Professor'].isin([0, 1, 2])] # Ensure target only has valid classes
# Add Belief_Label for plotting
replace_labels = {0: 'No', 1: 'Partially', 2: 'Yes'}
data['Belief_Label'] = data['AI_Fully_Replace_Professor'].map(replace_labels)
# Preserve original data for some plots
data_original = data.copy()
# Define features and target
features = ['Age', 'Education Level', 'AI_Teaching_Rating', 'Used_AI_Learning_Tools',
            'Best_Personalized_Learning', 'Lack of Emotional Intelligence',
            'Ethical Concerns', 'Limited Creativity', 'Risk of Bias in AI']
target = 'AI_Fully_Replace_Professor'
# Preprocessing: Separate categorical and numerical features
categorical_features = ['Age', 'Education Level', 'Best_Personalized_Learning']
numerical_features = ['AI_Teaching_Rating', 'Used_AI_Learning_Tools',
                      'Lack of Emotional Intelligence', 'Ethical Concerns',
                      'Limited Creativity', 'Risk of Bias in AI']
# Create a preprocessor for encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])
# Build a pipeline with preprocessing and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
])
# Split the data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) # For predicted belief scores
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nModel Performance on Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
# Cross-validation for robustness
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
print(f"\nCross-validated F1-score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
# Check class distribution
print("\nClass Distribution in Target:")
print(y.value_counts(normalize=True))
# --- Visualizations ---
# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Partially', 'Yes'],
            yticklabels=['No', 'Partially', 'Yes'],
            annot_kws={'size': 12, 'weight': 'bold'}, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('Predicted', fontsize=12, labelpad=10)
plt.ylabel('Actual', fontsize=12, labelpad=10)
plt.tight_layout(pad=2)
plt.show()
# 2. Class Distribution Bar Plot
plt.figure(figsize=(10, 6))
replace_counts = data['AI_Fully_Replace_Professor'].value_counts(normalize=True) * 100
replace_counts.index = replace_counts.index.map({0: 'No', 1: 'Partially', 2: 'Yes'})
bars = plt.bar(replace_counts.index, replace_counts, color=sns.color_palette('Set2')[0:3], edgecolor='black', linewidth=1.5)
plt.title('Belief in AI Replacing Professors', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('Opinion', fontsize=12, labelpad=10)
plt.ylabel('Percentage (%)', fontsize=12, labelpad=10)
plt.xticks(rotation=0, fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', fontsize=10, color='#333333', weight='bold')
plt.legend(bars, ['No', 'Partially', 'Yes'], title='Belief', loc='upper right', frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout(pad=2)
plt.show()
# 3. Feature Coefficients Bar Plot
coef = model.named_steps['classifier'].coef_ # Shape: (n_classes, n_features)
feature_names = preprocessor.get_feature_names_out()
n_classes = 3 # No, Partially, Yes
class_names = ['No', 'Partially', 'Yes']
fig, axes = plt.subplots(n_classes, 1, figsize=(12, 8), sharex=True)
for i, ax in enumerate(axes):
    coef_i = coef[i]
    sorted_idx = np.argsort(np.abs(coef_i))[-10:] # Top 10 features by magnitude
    colors = ['#FF6B6B' if x > 0 else '#45B7D1' for x in coef_i[sorted_idx]]
    bars = ax.barh(feature_names[sorted_idx], coef_i[sorted_idx], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title(f'Feature Coefficients for Class: {class_names[i]}', fontsize=12, weight='bold')
    ax.set_xlabel('Coefficient Value', fontsize=12, labelpad=10)
plt.tight_layout(pad=2)
plt.show()
# 4. Predicted vs Actual AI Replacement Belief (Scatter Plot)
class_values = np.array([0, 1, 2]) # Values for "No," "Partially," "Yes"
y_pred_belief = np.sum(y_pred_proba * class_values, axis=1) # Weighted sum: 0*P(0) + 1*P(1) + 2*P(2)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_test, y_pred_belief, c=y_test, cmap='viridis_r', alpha=0.8, edgecolors='black', linewidth=0.5, s=100, label='Data Points')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.colorbar(scatter, label='Actual Belief Score', shrink=0.8)
plt.title('Predicted vs Actual AI Replacement Belief', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('Actual Values', fontsize=12, labelpad=10)
plt.ylabel('Predicted Belief Values', fontsize=12, labelpad=10)
plt.xticks([0, 1, 2], ['No', 'Partially', 'Yes'])
plt.yticks([0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0])
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black')
plt.grid(True)
plt.tight_layout(pad=2)
plt.show()
# --- Additional Visualizations ---
# 5. Most Cited Challenges
plt.figure(figsize=(12, 6))
challenge_counts = data[challenges].sum()
bars = plt.bar(challenge_counts.index, challenge_counts, color=sns.color_palette('Blues_r', len(challenges)), edgecolor='black', linewidth=1.5)
plt.title('Most Cited Challenges to AI Replacing Professors', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('Challenges', fontsize=12, labelpad=10)
plt.ylabel('Number of Respondents', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, str(int(yval)), ha='center', fontsize=10, color='#333333', weight='bold')
plt.legend(bars, challenges, title='Challenges', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout(pad=2)
plt.show()
# 6. Correlation Matrix
plt.figure(figsize=(10, 6))
corr_features = data[['AI_Teaching_Rating', 'Used_AI_Learning_Tools'] + challenges]
correlation_matrix = corr_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', vmin=-1, vmax=1, center=0, square=True, linewidths=1,
            annot_kws={'size': 10, 'weight': 'bold'}, cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
plt.title('Correlation Matrix of Features', fontsize=16, weight='bold', pad=15, color='#333333')
plt.tight_layout(pad=2)
plt.show()
# 7. Belief vs. Field of Study
plt.figure(figsize=(14, 6))
sns.countplot(x='Field of Study', hue='Belief_Label', data=data, palette=sns.color_palette('Set2')[0:3],
              order=data['Field of Study'].value_counts().index, edgecolor='black', linewidth=1.5)
plt.title('Belief in AI Replacement by Field of Study', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('Field of Study', fontsize=12, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.legend(title='Belief', loc='upper right', frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout(pad=2)
plt.show()
# 8. AI Teaching Rating Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['AI_Teaching_Rating'], bins=5, color=sns.color_palette('Greens_r', 1)[0], edgecolor='black',
             linewidth=1.5, kde=True, line_kws={'color': '#9B59B6', 'linewidth': 2})
plt.title('AI Teaching Rating Distribution', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('AI Teaching Rating', fontsize=12, labelpad=10)
plt.ylabel('Count', fontsize=12, labelpad=10)
plt.xticks(fontsize=11)
plt.tight_layout(pad=2)
plt.show()
# 9. Most Used AI Tools
tools_series = data['AI_Tools_Used'].str.replace(' and ', ', ').str.split(', ').explode().str.strip()
tool_mapping = {
    'ChatGPT': ['ChatGPT', 'Chat gpt', 'Chat GPT', 'Chatgpt', 'chatgpt', 'CHAT GPT'],
    'Meta AI': ['Meta AI', 'meta', 'Meta'],
    'Gemini': ['Gemini'],
    'deepseek': ['deepseek']
}
for standard_name, variations in tool_mapping.items():
    tools_series = tools_series.replace(variations, standard_name)
allowed_tools = ['ChatGPT', 'Meta AI', 'Gemini', 'deepseek']
tools_series = tools_series[tools_series.isin(allowed_tools)]
top_tools = tools_series.value_counts()
plt.figure(figsize=(10, 6))
bars = plt.bar(top_tools.index, top_tools, color=sns.color_palette('Set2', len(top_tools)), edgecolor='black', linewidth=1.5)
plt.title('Most Used AI Tools', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('AI Tools', fontsize=12, labelpad=10)
plt.ylabel('Number of Respondents', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.legend(bars, top_tools.index, title='AI Tools', loc='upper right', frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout(pad=2)
plt.show()
# 10. Student Opinions on AI Role in Education
role_counts = data.get('AI_Role_in_Education', pd.Series()).value_counts(normalize=True) * 100
plt.figure(figsize=(10, 6))
bars = plt.bar(role_counts.index, role_counts, color=sns.color_palette('Pastel1', len(role_counts)), edgecolor='black', linewidth=1.5)
plt.title('Student Opinions on AI Role in Education', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('AI Role', fontsize=12, labelpad=10)
plt.ylabel('Percentage (%)', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    if not pd.isna(yval):
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', fontsize=10, color='#333333', weight='bold')
plt.legend(bars, role_counts.index, title='AI Role', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout(pad=2)
plt.show()
# 11. Best Approach for Personalized Learning
personalized_counts = data_original.get('Best_Personalized_Learning', pd.Series()).value_counts(normalize=True) * 100
plt.figure(figsize=(10, 6))
bars = plt.bar(personalized_counts.index, personalized_counts, color=sns.color_palette('Set2')[0:3], edgecolor='black', linewidth=1.5)
plt.title('Best Approach for Personalized Learning', fontsize=16, weight='bold', pad=15, color='#333333')
plt.xlabel('Approach', fontsize=12, labelpad=10)
plt.ylabel('Percentage (%)', fontsize=12, labelpad=10)
plt.xticks(rotation=0, fontsize=11)
for bar in bars:
    yval = bar.get_height()
    if not pd.isna(yval):
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', fontsize=10, color='#333333', weight='bold')
plt.legend(bars, personalized_counts.index, title='Approach', loc='upper right', frameon=True, facecolor='white', edgecolor='black')
plt.tight_layout(pad=2)
plt.show()
