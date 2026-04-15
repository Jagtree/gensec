CI, coverage and versioned documentation
========================================

Versioned documentation with sphinx-multiversion
------------------------------------------------

The HTML documentation is versioned using :mod:`sphinx_multiversion`.

Documentation versions are derived from Git **branches**, not from the
package version defined in ``pyproject.toml`` and not from patch-level
Git tags.

The project follows this rule:

* **Code** is versioned using semantic versions ``X.Y.Z`` (git tags, as usual).
* **Documentation** is versioned at the **minor level** ``X.Y``.

This avoids unnecessary documentation version bumps for patch-only
changes (including documentation fixes).

---

Documentation branches
~~~~~~~~~~~~~~~~~~~~~~

Each supported documentation version corresponds to a dedicated Git
branch:

* ``main``
  Represents the *latest* documentation.

* ``doc/X.Y``
  Represents the documentation for all ``X.Y.*`` releases.

For example::

    main        → latest documentation
    doc/0.1     → documentation for all 0.1.x releases

---

sphinx-multiversion configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The relevant configuration in ``conf.py`` is::

    # Build documentation only for branches
    smv_branch_whitelist = r'^(main|doc/\\d+\\.\\d+)$'

    # Explicitly disable tags
    smv_tag_whitelist = r"$^"

    # Mark "main" as the latest version
    smv_latest_version = "main"

With this setup:

* Documentation is built for ``main`` and all ``doc/X.Y`` branches.
* Git tags (for example ``v0.1.3`` or ``v0.1.4``) are **ignored**, but they affect
  SCRIPT version
* Documentation fixes do **not** require creating new Git tags.

---

Adding or updating a documentation version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add or maintain documentation for a minor release ``X.Y``:

1. Create a documentation branch (once per minor version)::

       git checkout main
       git checkout -b doc/0.1
       git push origin doc/0.1

2. Apply documentation fixes or updates on that branch::

       git checkout doc/0.1
       # edit .rst files
       git commit -am "Docs: update documentation for 0.1.x"

3. Rebuild the multi-version documentation::

       uv run sphinx-multiversion docs docs/_build/multiversion

This will create version-specific directories such as::

    docs/_build/multiversion/latest/
    docs/_build/multiversion/0.1/

---

Building into ``public/``
-------------------------

For GitLab and GitHub Pages it is convenient to build the documentation directly
into the ``public/`` directory. This is the directory that GitLab and GitHub
commonly serve as the root of the project site.

Locally you can run::

    uv run sphinx-multiversion docs public

This creates::

    public/latest/
    public/0.1/

GitLab and GitHub Pages requires a ``public/index.html`` file at the root level.
Since ``sphinx-multiversion`` creates only version-specific
subdirectories, a redirect file must be added.

The redirect ensures that visitors accessing the root of the GitLab and GitHub
Pages site are automatically sent to the *latest* documentation.

---

Removing old documentation versions
-----------------------------------

Simply deleting a directory under ``docs/_build/multiversion/`` or
``public/`` is temporary. The next time :mod:`sphinx_multiversion` runs,
it will recreate all versions for which a matching Git branch exists.

To permanently remove an old documentation version:

* Delete the corresponding ``doc/X.Y`` branch::

      git branch -d doc/0.0
      git push origin --delete doc/0.0

* Or exclude it by tightening ``smv_branch_whitelist``.

After updating branches or configuration, rerun
``sphinx-multiversion`` to regenerate the documentation output.
