```
Project Structure

ragify/
│
├── data/                      # Place your PDF files here
│   └── yourfile.pdf
│
├── faiss_index/              # Will be created automatically after vector creation
│
├── app.py                    # Your main Streamlit script
└── requirements.txt          # (optional) dependencies
```
# use vs code /any ide
# create a folder ragify/provide your own name 
# Step 1- Create your virtual environment and install all dependencies given below 

```
pip install streamlit boto3 langchain numpy
pip install pypdf

```
# Step 2- Create Access ID in your AWS IAM Account
# Step 3- Open local CMD and run this
``` aws configure ```

# Step 4- Provide the Access Credentials in the CMD
# Step 5- return to ide and run the command in terminal
```
 streamlit run app.py
```
