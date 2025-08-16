# Evaluations

## Personal Experience

WHile teh project was very interesting to work on, and I tried to provide the best result as I could, I perhaps was not able to fully understand the need for a RAG linking DriveLM and NuScenes.

Perhaps the Mini version of the NuScenes does not encompass any of the datapoints present in DriveLM, both the whole and the train_sample, because of which I was not able to link any of the datapoints from one dataset to another. I tried to link them through images, and that too was not enough.

What would have been fruitful is if I was given the dataset to work with, as it was not possible for me to download everything and check which datapoints matched. Perhaps a truncated version of the datasets, where there is clear matching would have been fruitful.

### What I tried to do

Since I was not able to match the datapoints, I first indexed the NuScenes-mini images into FAISS using [OpenAI's `CLIP ViT Large` checkpoint](https://huggingface.co/openai/clip-vit-large-patch14). I used the scene descitions from the DriveLM dataset as the query, and I obtained the relevant images by performing vector search on FAISS using this query. This is then used as the relevant image with the metadata that we need. I then passed these images along with the metadata and the actual question to the LM, and obtained results.

Since I was not sure about whether this solution was adequate, and because of limited time, I was only able to perform cosine similarity on a very small dataset. I wouldv'e loved to explore this dataset more, but just trying to link the two datasets together took me more than 3 days, and so I was not able to see what actually works and what does not. Unfortunately, I was not able to completely perform the evaluation tasks completely, which actually would've shown the true potential of my solution. However, my solution does work quite well on low resource settings, where we need images given a query.

For finetuning, I had lots of ideas in mind. I wanted to finetune the retriever user a contrastive learning technique, using a two-tower architecture, where we have one LM (like ModernBERT) for the query, and a VLM (like QWEN) for the images. A normal siamese architecture using just one single encoder would also be an idea worth exploring. Additionally, we could represent the image encodings as nodes in a graphs, where we connect the image encoding belonging to the same scenes in this graph, and apply graph learning here. I do not think I would need to finetune the Genrator LM. 

