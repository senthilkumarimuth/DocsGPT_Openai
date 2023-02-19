<h1 align="center">
  DocsGPT with Openai
</h1>

This application is for getting answer based on document, similar to question-answer chatbot. This application uses OpenAI Embedding to create contexts from document and later the context is prepend to query and the query is sent OpenAI's compltion API for getting answer

# How to create vector for your own document and use the application?

Git clone the application in your local. Run the below command in command prompt.

`git clone https://github/senthilkumarimuth/DocsGPT_Openai.git`
 
Install dependencies. cd to application folder and run the below command

`pip install -r requirements.text`
 
Go to scripts folder and run the script prepare_contextvector.py

`python prepare_contextvector.py`

Enter the document’s name for which you need context vector as in the example below

![Alt text](/readme_fiels/b3a8d398-f17d-419e-a607-eccb6f3dfcd3.png?raw=true "enter document name")

Note: here the document name is pdp.

Once run is complete, you will see pkl files generated at the directory 'vectorstore' as in the example below.

![vectorStoret](./readme_fiels/vecterstore.png)

<p align="center">
  <img src="./readme_fiels/vecterstore.png" width="350" title="vectorstore">
</p>


If you have reached this stage, you are successful at creating context vector for you document.

Next prepare prompts and keep in respective folder as shown below.

![Alt text](./readme_fiels/56c8dba5-452b-45b9-a9df-6ab5f4e8af3a.png?raw=true "prompt")

Now you are all setup to run the Flask application.


python app.py
Now visit to http://127.0.0.1:5000/ 

Choose the document to which you would like to chat.

![Alt text](./readme_fiels/206d5169-f176-4005-a23a-fa1616db744f.png?raw=true "web ui")

Now you are all setup to chat with the document!

 
