# Gizmo
> Your Local Cli Assistant
Gizmo is your local ai assistant. Claude code is at over 33,000 ✩ but not a bit is local. Gizmo is a completely-local, mcp-enabled, personal assistant that can even talk! Just come with good hardware and some MCP servers. Youll be able to tell it to turn off your light in no time!

## Installing / Getting started

Installing is easy. Just install [Ollama](https://ollama.com/),
and clone the repo!

```shell
git clone https://github.com/tacocatCLAUS/gizmo.git
cd gizmo
python setup.py
python gizmo.py
```

This will install everything necessary and have you set the configuration.

## Developing

If you want to help with development first of all THANK YOU! But all you need to do is install the project as shown above.
Running setup.py will install from model/requirements.txt but if you want to do this yourself just run: 
```shell
pip install -r model/requirements.txt
```

## Features

What's all the bells and whistles this project can perform?
* What's the main functionality
* You can also do another thing
* If you get really randy, you can even do this

## Configuration

Here you should write what are all of the configurations a user can enter when
using the project.

```shell
{
  "openai": false,
  "openai_model": "gpt-3.5-turbo",
  "openai_api_key": "",
  "hc": false,
  "hc_model": "meta-llama/llama-4-maverick-17b-128e-instruct",
  "devmode": false,
  "db_clear": true,
  "use_mcp": true,
  "voice": false,
  "rag_model": "none"
}
```
<img width="1360" height="820" alt="carbon" src="https://github.com/user-attachments/assets/299e47b4-255c-4af7-bbb9-915ec94e23f5" />


#### openai
Type: `String`  
Default: `'default value'`

State what an argument does and how you can use it. If needed, you can provide
an example below.

Example:
```bash
awesome-project "Some other value"  # Prints "You're nailing this readme!"
```

#### Argument 2
Type: `Number|Boolean`  
Default: 100

Copy-paste as many of these as you need.

## Contributing

When you publish something open source, one of the greatest motivations is that
anyone can just jump in and start contributing to your project.

These paragraphs are meant to welcome those kind souls to feel that they are
needed. You should state something like:

"If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome."

If there's anything else the developer needs to know (e.g. the code style
guide), you should link it here. If there's a lot of things to take into
consideration, it is common to separate this section to its own file called
`CONTRIBUTING.md` (or similar). If so, you should say that it exists here.

## Links

Even though this information can be found inside the project on machine-readable
format like in a .json file, it's good to include a summary of most useful
links to humans using your project. You can include links like:

- Project homepage: https://your.github.com/awesome-project/
- Repository: https://github.com/your/awesome-project/
- Issue tracker: https://github.com/your/awesome-project/issues
  - In case of sensitive bugs like security vulnerabilities, please contact
    my@email.com directly instead of using issue tracker. We value your effort
    to improve the security and privacy of this project!
- Related projects:
  - Your other project: https://github.com/your/other-project/
  - Someone else's project: https://github.com/someones/awesome-project/


## Licensing

One really important part: Give your project a proper license. Here you should
state what the license is and how to find the text version of the license.
Something like:

"The code in this project is licensed under MIT license."
