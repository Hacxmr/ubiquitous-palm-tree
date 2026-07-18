FASHION_PROMPT = """
You are an expert fashion analyst.

Analyze the image.

Return ONLY valid JSON.

Schema:

{
  "people": int,

  "upper_garment": {
    "type":"",
    "color":"",
    "pattern":"",
    "material":""
  },

  "lower_garment": {
    "type":"",
    "color":"",
    "pattern":""
  },

  "outerwear": {
    "type":"",
    "color":""
  },

  "footwear": {
    "type":"",
    "color":""
  },

  "accessories":[
  ],

  "style":"",

  "scene":"",

  "activity":"",

  "weather":"",

  "dominant_colors":[]
}

Return JSON only.
"""