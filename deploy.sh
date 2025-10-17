#!/bin/bash

git worktree add --force _deploy gh-pages
find _deploy -mindepth 1 -not -name '.git' -exec rm -rf {} +
cp -r _site/* _deploy/

cd _deploy
git add --all
git commit -m "Deploy site to gh-pages"
git push origin gh-pages --force

cd ..
git worktree remove _deploy
