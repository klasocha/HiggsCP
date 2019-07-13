FROM rootproject/root-ubuntu16
RUN sudo apt-get update
RUN sudo apt-get install -y python-pip fish virtualenv vim wget 
RUN mkdir /home/builder/.venv
RUN virtualenv /home/builder/.venv/higgs
COPY --chown=builder:builder . /home/builder/project
RUN sudo bash -c 'echo "builder ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers'
ENTRYPOINT /usr/bin/fish
