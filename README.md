# Visual-Question-Answering-using-Multimodal-Transformers


## Project Overview
This project aimed to develop an advanced Visual Question Answering (VQA) system that seamlessly integrates textual and visual data to provide precise answers to image-based questions. The system leverages **BERT** for text embeddings and **Vision Transformer (ViT)** for visual feature extraction, showcasing the potential of multimodal AI in addressing complex queries. The project involved extensive data preprocessing, model training, integration, and deployment, demonstrating my proficiency in working with large-scale AI systems.

---

## Technologies Used
- **Programming Language:** Python  
- **Libraries and Tools:** PyTorch, OpenCV, Hugging Face Transformers, Streamlit, AWS EC2, NumPy, Pandas  
- **Models:** BERT for text embeddings, Vision Transformer (ViT) for visual embeddings  
- **Optimization Techniques:** Adam Optimizer, Cross-Entropy Loss, Grid Search for Hyperparameter Tuning, Data Augmentation  

---

## Dataset Preparation
The dataset preparation involved handling a large volume of textual and visual data. Images were processed using **OpenCV**, which included resizing images to 224x224 pixels, normalizing pixel values for consistency, and applying augmentation techniques like rotation, flipping, and brightness adjustments to enhance diversity and improve model robustness.

For textual data, tokenization was performed using **Hugging Face's Tokenizers**, breaking down complex questions into subword units compatible with **BERT**. Below is an example code snippet illustrating the tokenization process:
python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
question = "What objects are present in the image?"
tokens = tokenizer.tokenize(question)

This ensured that the textual input was efficiently processed for embedding generation.

---

## Workflow
### 1. Textual Data Pipeline
The textual data pipeline utilized **BERT** to convert questions into dense vector embeddings that captured the semantic essence:
python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(question, return_tensors='pt')
outputs = model(**inputs)
text_embedding = outputs.last_hidden_state.mean(dim=1)

This embedding process ensured that the textual inputs were transformed into numerical vectors for analysis.

### 2. Visual Data Pipeline
Images were processed through **ViT** to generate embeddings representing visual features:
python
from transformers import ViTModel

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
image = preprocess_image(image_path)
outputs = vit_model(image)
visual_embedding = outputs.last_hidden_state.mean(dim=1)

This allowed the model to understand and encode visual elements effectively.

### 3. Fusion Technique
The **late fusion technique** was employed to integrate the textual and visual embeddings at a later stage, enhancing the system's ability to generate accurate answers:
python
import torch

combined_embedding = torch.cat((text_embedding, visual_embedding), dim=1)

This fusion ensured that both modalities contributed equally to the final prediction.

### 4. Training Process
The model was trained using the **Adam optimizer** for efficient gradient descent and **cross-entropy loss** for handling multiclass classification problems:
python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

The training loop iterated over batches, adjusting weights to minimize loss.

---

## Key Functions and Methods
- **Data Preprocessing:** OpenCV for image manipulation, Hugging Face Tokenizer for text tokenization  
- **Model Training:** PyTorch with custom DataLoaders for efficient batch processing  
- **Embedding Integration:** Late fusion method for combining textual and visual embeddings  
- **Evaluation:** WUPS score for measuring semantic similarity between predicted and ground-truth answers  

---

## Deployment
The system was deployed using **Streamlit** for an interactive user interface, allowing users to upload images and input questions. The backend was hosted on **AWS EC2** for scalable real-time processing, ensuring that the system could handle multiple requests simultaneously.

---

## Challenges and Solutions
- **Aligning embeddings from different modalities:** This was solved using the late fusion technique, ensuring that both textual and visual inputs were represented accurately.  
- **Optimizing model performance:** Hyperparameter tuning through **Grid Search** was performed to find the best combination of learning rates, batch sizes, and dropout rates.  
- **Handling large datasets:** Efficient data loading and augmentation techniques were implemented to manage extensive training data.

---

## Results and Insights
The VQA system achieved high semantic accuracy, validated using the **WUPS score**, which measures the semantic similarity between predicted and actual answers. This project highlighted the strength of **transformer architectures** in multimodal tasks, emphasizing the importance of integrating visual and textual data for comprehensive AI solutions.

This project underscores my expertise in AI model development, multimodal data integration, and efficient workflow optimization, showcasing my ability to handle end-to-end AI solutions. 
