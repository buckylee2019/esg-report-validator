# watsonx-esg-validator

## Setup

`docker build . -t <YOUR IMAGE PREFER NAME>`
`docker run -it -d -v <this repo>:/app -p 8501:8501 <YOUR IMAGE PREFER NAME>`


## Open Browser 

1. Enter localhost:8501 in any preferred browser.
![Streamlit](image/ui.png)

2. Upload your ESG Report
![Upload File](image/upload.png)

3. Check sidebar
4. Select Model, highly recommend llama2-70b-chat for Chinese Data
5. Click “自動生成”!
![Generate](image/generate.png)