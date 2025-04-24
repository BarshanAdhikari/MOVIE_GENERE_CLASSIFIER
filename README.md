
# 🎬 Movie Genre Classification

This project focuses on classifying movies into their respective genres based on their descriptions using machine learning techniques. It includes data preprocessing, feature engineering, and applying a machine learning model to predict genres.

---

## 📌 Step-by-Step Breakdown

### 1. **Import Libraries**
Essential libraries like `pandas`, `numpy`, `nltk`, `sklearn`, and visualization tools like `matplotlib` and `seaborn` are imported.

---

### 2. **Load Dataset**
```python
train_data = pd.read_csv("train_data.txt", sep=":::", names=['Title', 'Genre', 'Description'], engine='python')
```
The dataset contains movie titles, genres, and descriptions.

---

### 3. **Explore Data**
- Checked for unique genres.
- Counted occurrences of each genre.
- Explored the dataset structure using `.info()` and `.head()`.

---

### 4. **Preprocess Descriptions**
- **Remove Punctuation**
- **Convert to Lowercase**
- **Tokenization** using NLTK
- **Stemming** using `PorterStemmer`

These steps prepare text data for vectorization.

---

### 5. **TF-IDF Vectorization**
Text data is transformed into numerical format using `TfidfVectorizer`, turning each description into a vector based on word importance.

---

### 6. **Feature Engineering**
- Calculated the length of each description.
- Normalized this feature using `MinMaxScaler`.

---

### 7. **Train-Test Split**
Divided the data into training and testing sets to evaluate model performance.

---

### 8. **Model Training**
A **Random Forest Classifier** is used to train the model on the TF-IDF features and the scaled length of the description.

---

### 9. **Evaluation Metrics**
- **Accuracy**
- **Precision**
- **Recall**
- **Confusion Matrix**

These metrics help assess how well the model performs genre classification.

---

## ✅ Requirements

- Python 3.x
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn

---

## 🚀 Run the Project

1. Install dependencies.
2. Ensure `train_data.txt` is in the correct path.
3. Run all cells in the notebook.
