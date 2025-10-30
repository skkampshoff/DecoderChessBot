# Bot specification

All chess bots *must* be submitted in the format of a python module. If you are unsure where to start, check out the demo bots.
Feel free to start by modifying their code.

The bot module must take the following two options as command line arguments:

* `train`
* `play`

If `play` is selected, then assigned team color must be given as the very next argument, either `w` or `b`.

For example:

```bash
python -m bot_module_name train
python -m bot_module_name play w
python -m bot_module_name play b
```

Your bot must be able to handle being invoked in each of the above 3 options.

