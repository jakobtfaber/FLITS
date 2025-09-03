Contributing
============

1. Fork the repository and create a feature branch.
2. Ensure new code includes NumPy-style docstrings and unit tests where
   possible.
3. Run the linters and documentation build before submitting a pull
   request:

   .. code-block:: bash

      python -m py_compile $(git ls-files '*.py')
      sphinx-build -b html docs docs/_build/html

4. Open a pull request describing your changes.
