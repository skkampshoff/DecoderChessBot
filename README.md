## AI Chess Competition: Participant Guide

Welcome to the AI Chess Competition! This guide tells you the rules, resources, and technicalities for developing and submitting your AI chess engine. The final competition will take place at [Nanocon](https://sites.google.com/view/dsunanocon/events/sunday?authuser=0)!

## 1. General Rules
* Team Size: 1-3 people
* Signup: [this google form](https://forms.gle/eJiDRibwwXobutfSA)
* Contact: for any questions, issues, or clarifications, reach out to one of the AI Club officers or Eddie French on Discord
* Do not just clone GitHub repositories or do any form of plagiarism!

## 2. Materials Provided
Sample chess bot:
* A Python module will be provided containing boilerplate code and a working (not necessarily 'smart') sample bot.
Accessory materials:
* requirements.txt file (with the allowed python libraries)
* Chess Rules PDF will be provided for reference on standard rules of play (e.g., castling, en passant, draw conditions)
* Supplementary **videos** and an **intro to allowed libraries** may be provided to help new participants get started.
Other resources we suggest you take advantage of:
* [Simple Opening Database](https://www.kaggle.com/datasets/alexandrelemercier/all-chess-openings?select=openings_fen7.csv) (covering opening moves and/or end moves) to integrate into your engine.
* [Simple Chess Database](https://www.kaggle.com/datasets/datasnaek/chess?select=games.csv) for training or reference
* These files will be provided with the sample bot (see bot-spec.md for details), so you won't be required to download them yourself at the expense of training time
* You may use any other publicly available dataset, but your training script must handle downloading them, which counts against alloted training time.
* You may use the competition moderator to test your bot against the demo bots. To run with output in terminal, run `python -m compeition_moderator /path/to/white/bot /path/to/black/bot`. To run with graphical output, run `./visualize.sh /path/to/white/bot /path/to/black/bot`. Note: if running with gui, press `f` to toggle fullscreen.

## 3. Submission Guidelines
* All submissions must be written in Python
* The only allowed libraries:
    * `numpy`
    * `scikit-learn`
    * `keras`
    * `Pytorch`
    * `tensorflow`
    * `pandas`
    * `python-chess`
* No multiprocessing/multithreading please!
* Submission deadline: **Thursday, November 6th, 11:59 pm**
* Submission format: submit zipped folder containing a python module that complies with the bot specification (see bot-spec.md) via Discord
* Final Competition Date: **Sunday, November 9th, 10:00 am**, at the Dakota Playhouse

## 4. Engine Parameters and Training
* **Chess Engine:** You need to implement a functional **chess engine** using the allowed libraries. This can be neural, deterministic, or a mix of both!
* **I/O** Use the INPUT and OUTPUT commands we supply for io (see bot spec). Both will be strings in SAN form.
* **Hyperparameters:** Be prepared to detail and tune your model's **Hyperparameters**.
* **Depth:** The **search depth** of your engine is a critical factor and must be optimized within the time constraints.
* **Handling Training:** Your submission script will be run on the specified hardware. Ensure your training process is integrated in a reasonable fashion with the game playing script, or your model is pre-trained by the training script then saved in a manner that the game playing script can use.
* **Handling Inference:** Your script must be optimized for fast and efficient **inference** during the match.

## 5. Hardware and Time Constraints
* **Hardware Specs:**
* AMD Ryzen 9 5950X @ 5.09 GHz
* NVIDIA RTX 3090
* Persistent Memory:
* Limit - 20GB disk space used per bot (data on disk which persists between training and playing)
* Totals:
* 64 GB RAM
* 24 GB VRAM
* Runtime Limits:
* 25 GB RAM at any time PER PLAYER
* 10 GB VRAM at any time PER PLAYER
* **Time Limit:** 5 minutes per side to make ALL moves (similar to human time controls). That is, you have a 5 minute timer that counts down when it is your turn and stops when you submit your move. When you run out of time, you lose.

## 6. After Competition Period
* After the competition concludes, we encourage teams to prepare and deliver a presentation on methodology, explaining your algorithm, training process, and time management strategies.
* **Award:** The winning team will receive a **Raspberry Pi 4 kit**!

**Have fun!**
