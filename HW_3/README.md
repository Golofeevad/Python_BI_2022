# Homework 3
## _Ultraviolence_
Mikhail wanted his paper readers to suffer. I suffered and created this instruction for launch ultraviolence from terminal. 
My Ubuntu version Ubuntu 22.04.1 LTS.

***Python version***

Ultraviolence.py file requires python version 3.11. We can install only prerelease: Python 3.11.0rc2. Tipe following commans:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo apt install python3.11-venv
sudo apt install python3.11-dev
```
***Required packages***

-Create folder with ultraviolence.py and _requirements.txt_ file from this repository.

-Firstly, create virtual environment in folder with project using command
`python3.11 -m venv environment`

-Work from this environment:
`source environment/bin/activate`

-Install requirements from _requirements.txt_ to virtual environment.
`pip install -r requirements.txt`

-Then it's time for ultraviolence!
`python3 ultraviolence.py`




