# set the base image 
FROM python:3.7

# File Author / Maintainer
MAINTAINER timber_smallsea

#add project files to the usr/src/app folder
ADD . /usr/src/app/ye_soft

#set directoty where CMD will execute 
COPY requirements.txt ./
WORKDIR /usr/src/app/ye_soft

# Get pip to download and install requirements:
RUN pip install --upgrade --default-timeout=10000 --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

# Expose ports
EXPOSE 3502

# default command to execute
CMD [ "python", "-u", "manage.py", "runserver", "0.0.0.0:3502" ]