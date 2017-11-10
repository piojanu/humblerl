# Hello!

I'm really glad you're reading this, we value people interested in Deep Learning and Reinforcement Learning!
*You should also familiraze with our values. Read CODE_OF_CONDUCT.md in root directory of this repo.*

# What contribution do we need?

From *reviewing PR-s and proposing some ideas* in issues to *coding*. Start small and then build a momentum. Also don't be intimidated if you've just started working in open-source, everybody was in your position at the beginning :) Learn with each commit/contribution!

# How to start?

Read README.md in root directory of this repo. It's best starting point.

# _(Not only)_ Coding standards

* **Python**

    [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) is in operation.
    
    If you are emacs user, I recommend installing this package: py-autopep8. Configuration:  
    ```elisp
    ;; enable autopep8 formatting on save
    (require 'py-autopep8)
    (add-hook 'elpy-mode-hook 'py-autopep8-enable-on-save)
    ```  
    If you look for the best python/markdown/everything IDE and want to configure it easily, here is a guide for you: https://realpython.com/blog/python/emacs-the-best-python-editor/ and then http://jblevins.org/projects/markdown-mode/ .

* **Git commits**

    * [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) is in operation.

    * If you work in this repo, remote branch names should follow this template:
        
		`dev/<user name>/<your branch name>`.
                
* **Pull requests**

    * If you want to commit to this repo: fork it, work locally and then create a pull request.  
    **Pull request to master is mandatory even for collaborators!**
    
    * Also before creating pull request, squash your commits. It provides clarity in master branch history.
    
    * Keep PR-s small! It's easier to do code-review on them and everybody loves small incremetal changes ;)
