from openai import OpenAI
import utils

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = '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'



client = OpenAI(api_key=api_key)
#client.api_key = api_key

input_instruction = "I need the dialogue broken into logical scenes. The scene should begin with [Scene] in its own line and can have [Scene: %description_placeholder%] where description is one sentence about the scene - such as who is in the scene. Add [Scene] where a new scene should begin. \
In some places instead of [Scene: %description_placeholder%], there might be scene description in round brackets (description), convert these to [Scene: %description_placeholder%] . There might also be comments within () that are not indicators of a new scene. \
If there is already [Scene] at start of a scene, dont add another. \
Do not remove any lines. \
Respond with only output text and no other comments so that the output can be fed into a file. \
The output should be in the same format as the input, with the same number of lines."

root_data_dir = utils.get_data_root_dir()
input_episode = 'friends_s02e11'

file_name = f"{root_data_dir}/stimuli/transcripts/friends/full/{input_episode}.txt"
with open(file_name, 'r', encoding='utf-8') as file:
    lines = file.readlines()

input_text = '\n'.join(lines)

response = client.responses.create(
    model="gpt-4.1",
    # input="Write a one-sentence bedtime story about a unicorn.",
    input=[
        {
            "role": "system", 
            "content": [
        {
          "type": "input_text",
          "text": input_instruction,
        }
      ]
        },
        {"role": "user", "content": [
        {
          "type": "input_text",
          "text": input_text,
        }
      ]}
    ]
)
resp = response.output_text
resp = resp.split('\n')
print('lines', len(lines), len(resp))

# For original text (strip newlines from each line)
original_chars = ''.join(line.rstrip('\n') for line in lines)

# For response (already split by '\n' so no newlines)
response_chars = ''.join(resp)

print(f"Original chars: {len(original_chars)}, Response chars: {len(response_chars)}")
if len(original_chars) < len(response_chars):
    print(f'***ERROR: Original chars and response chars are not the same*** {input_episode}')

output_file_name = f"{root_data_dir}/stimuli/transcripts/friends/full/{input_episode}_scenes.txt"
with open(output_file_name, 'w', encoding='utf-8') as file:
    for line in resp:
        file.write(line + '\n')
#print(response.output_text)

# I need the dialogue broken into logical scenes. The scene should begin with [Scene] in its own line and can have [Scene: description] where description is one sentence about the scene - such as who is in the scene. Add [Scene] where a new scene should begin.
# In some places instead of [Scene: description], there might be scene description in round brackets (description), convert these to [Scene: description] . There might also be comments within () that are not indicators of a new scene.

# Respond with only output text and no other comments so that the output can be fed into a file."