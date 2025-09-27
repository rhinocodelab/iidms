prompt = """
You are an intelligent document classifier and summarizer.

Here is a list of existing categories:
{existing_categories}

Read the following document and perform the following tasks:

1. Carefully determine the most appropriate category for this document as "document_type". 
   - If an existing category is a **strong match**, use it.
   - If none of the categories fit **clearly and specifically**, create a new, short category name that better describes the document.
2. Identify 3-5 major tags or keywords from the document content as "Tags".
3. Provide a 2-3 sentence summary of the document as "summary".
4. Explain why you chose the document_type in a field called "reasoning".

Use this JSON format for the output:

{
"document_type": "<Document Type>",
"Tags": "<Major tags present in the document>",
"summary": "<Summary of the content>",
"reasoning": "<Brief classification reasoning>"
}

-------------------------
{content}
-------------------------

Please classify the document and return the result as a JSON object only.
Do not generate intermediate steps or explanation.
"""