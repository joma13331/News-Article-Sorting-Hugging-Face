# GIt Commands Used

1. git init : To initialize the Version Control using Git
```
    git init
```

2. git pull: to pull the updated code from the repository
```
git pull <The_address_of_the_repository>
```

3. git add
```
git add <filename1> <filename2>
```
OR (what i usually use to add all the files)
```
git add .
```

4. git commit: to store all the changes to git history
```
git commit -m "<message>"
```

5. git remote add: to store the repository address to a variable
```
git remote add <variable_name> <repository address>
```

6. git branch -m: to rename a branch
```
git branch -m <old_branch_name> <new_branch_name>
```

7. git remote: to check the repository address that are stored in variables variable
```
git remote -v
```

8. git branch: to check all available branches in the git project
```
git branch
```

9. git branch <branch_name>: create a new branch with name branch_name
```
git branch <branch_name>
```

10. git checkout <branch_Name>: Switch to git branch with name branch_name
```
git checkout <branch_name>
```

# Docker Commands used

1. BUILD DOCKER IMAGE
```
docker build -t <image_name>:<tagname> .
```
> Note: Image name for docker must be lowercase


2. To list docker image
```
docker images
```

3. Run docker image
```
docker run -p 5000:5000 -e PORT=5000 f8c749e73678
```

4. To check running container in docker
```
docker ps
```

5. To stop docker conatiner
```
docker stop <container_id>
```