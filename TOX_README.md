Testing env
===========

Test should be created in ./test directory. It's worth to note, that each folder
in ./test directory should have \_\_init\_\_.py file to be properly recognise.

Tests should have names beginning with test, for example test\_anal\_utils.py
Test file consists of classes with names beginning with Test, each have methods
named accordingly (also with test prefix).

Prepare testing utils
---------------------

To prepare environment tox package needs to be installed with command

```bash
  conda install -c conda-forge tox
```
if conda is used or just


```bash
  pip install tox
```
when bare virtualenv is used.

Important note is to put name of package you introducing into project to
requirements.txt, and then run `tox --recreate`.

Running tests
-------------

Just type
```bash
  tox
```

If something crushes (not tests) try with tox --recreate.

TODO
----

Add some linters to ensure code quality
