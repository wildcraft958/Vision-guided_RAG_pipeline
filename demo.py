from google import genai
from google.genai import types
import pathlib
import httpx
from PROMT_TEMPLATE import get_PROMPT_TEMPLATE

client = genai.Client(api_key="AIzaSyD4L8hVk3mNjDVl3hfAv9SU7cpvWubEjLw")

# Retrieve and encode the PDF byte
filepath = pathlib.Path('./examples/sample_pdfs/0512222v2_copy.pdf')

prompt = get_PROMPT_TEMPLATE() 

response = client.models.generate_content(
  model="gemini-2.5-pro",
  contents=[
      types.Part.from_bytes(
        data=filepath.read_bytes(),
        mime_type='application/pdf',
      ),
      prompt])

output_file = pathlib.Path('sample_outputs/output4.txt')
output_file.parent.mkdir(exist_ok=True) 
output_file.write_text(response.text)

if __name__ == '__main__':
    pass
