FROM ubuntu:24.10

#? ---- . ---- ---- ---- ---- . ----
#? ARGs:
#? - all ARG_***_HOST args must encode pahts in the host
#? - all ARG_***_CNTR args must encode pahts in the container
#? - all ARG_***_HOST paths should be relative to this dockerfile, unsless specyfied otherwise
#? - all ARG_***_CNTR paths must be absolute
#? - ARG_***_HOST directory paths should never end with a slash "/"
#? - ARG_***_CNTR directory paths should never end with a slash "/"
#? ---- . ---- ---- ---- ---- . ----

# app analysis's general:
ARG ARG_APP_ANALYSIS_DOCKERFILE_NAME
ARG ARG_APP_ANALYSIS_TIMEZONE
ARG ARG_APP_ANALYSIS_LOCALE
ARG ARG_APP_ANALYSIS_WORKDIR_HOST
ARG ARG_APP_ANALYSIS_WORKDIR_CNTR
ARG ARG_APP_ANALYSIS_CONFIGDIR_HOST=".${ARG_APP_ANALYSIS_DOCKERFILE_NAME}/.config"
ARG ARG_APP_ANALYSIS_CONFIGDIR_CNTR="/.${ARG_APP_ANALYSIS_DOCKERFILE_NAME}/.config"
ARG ARG_APP_ANALYSIS_TEMPLATEDIR_HOST=".${ARG_APP_ANALYSIS_DOCKERFILE_NAME}/.templates"
ARG ARG_APP_ANALYSIS_TEMPLATEDIR_CNTR="/.${ARG_APP_ANALYSIS_DOCKERFILE_NAME}/.templates"
ARG ARG_APP_ANALYSIS_RESULTSDIR_HOST
ARG ARG_APP_ANALYSIS_RESULTSDIR_CNTR

# app analysis's python3:
ARG ARG_APP_ANALYSIS_PYTHON3_VERSION="3.12"
ARG ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR="/opt/python/${ARG_APP_ANALYSIS_PYTHON3_VERSION}/venv"

# app analysis's node:
ARG ARG_APP_ANALYSIS_NODE_VERSION="22"

# print args:
RUN echo ""\
    && echo "app analysis's general:"\
    && echo "- ARG_APP_ANALYSIS_DOCKERFILE_NAME: ..... ${ARG_APP_ANALYSIS_DOCKERFILE_NAME}"\
    && echo "- ARG_APP_ANALYSIS_TIMEZONE: ............ ${ARG_APP_ANALYSIS_TIMEZONE}"\
    && echo "- ARG_APP_ANALYSIS_LOCALE: .............. ${ARG_APP_ANALYSIS_LOCALE}"\
    && echo "- ARG_APP_ANALYSIS_WORKDIR_HOST: ........ ${ARG_APP_ANALYSIS_WORKDIR_HOST}"\
    && echo "- ARG_APP_ANALYSIS_WORKDIR_CNTR: ........ ${ARG_APP_ANALYSIS_WORKDIR_CNTR}"\
    && echo "- ARG_APP_ANALYSIS_CONFIGDIR_HOST: ...... ${ARG_APP_ANALYSIS_CONFIGDIR_HOST}"\
    && echo "- ARG_APP_ANALYSIS_CONFIGDIR_CNTR: ...... ${ARG_APP_ANALYSIS_CONFIGDIR_CNTR}"\
    && echo "- ARG_APP_ANALYSIS_TEMPLATEDIR_HOST: .... ${ARG_APP_ANALYSIS_TEMPLATEDIR_HOST}"\
    && echo "- ARG_APP_ANALYSIS_TEMPLATEDIR_CNTR: .... ${ARG_APP_ANALYSIS_TEMPLATEDIR_CNTR}"\
    && echo "- ARG_APP_ANALYSIS_RESULTSDIR_HOST: ..... ${ARG_APP_ANALYSIS_RESULTSDIR_HOST}"\
    && echo "- ARG_APP_ANALYSIS_RESULTSDIR_CNTR: ..... ${ARG_APP_ANALYSIS_RESULTSDIR_CNTR}"\
    && echo ""\
    && echo "app analysis's python3:"\
    && echo "- ARG_APP_ANALYSIS_PYTHON3_VERSION: ..... ${ARG_APP_ANALYSIS_PYTHON3_VERSION}"\
    && echo "- ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR: ... ${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}"\
    && echo ""\
    && echo "app analysis's node:"\
    && echo "- ARG_APP_ANALYSIS_NODE_VERSION: ........ ${ARG_APP_ANALYSIS_NODE_VERSION}"\
    && echo ""

#? ---- . ---- ---- ---- ---- . ----
#? proper:
#? - workdir:
#? ---- . ---- ---- ---- ---- . ----

# WORKDIR:
WORKDIR "${ARG_APP_ANALYSIS_WORKDIR_CNTR}"

#? ---- . ---- ---- ---- ---- . ----
#? proper:
#? - general setup:
#? ---- . ---- ---- ---- ---- . ----

# mark this dockerfile as an unattended script (mostly for apt-get):
# - see https://askubuntu.com/a/972528
ENV DEBIAN_FRONTEND=noninteractive

# config timezone:
RUN ln -snf /usr/share/zoneinfo/${ARG_APP_ANALYSIS_TIMEZONE} /etc/localtime && echo ${ARG_APP_ANALYSIS_TIMEZONE} > /etc/timezone

# mkdir /etc/apt/keyrings:
RUN mkdir -p "/etc/apt/keyrings"

# mkdir "${ARG_APP_ANALYSIS_CONFIGDIR_CNTR}":
RUN mkdir -p "${ARG_APP_ANALYSIS_CONFIGDIR_CNTR}"

# mkdir "${ARG_APP_ANALYSIS_TEMPLATEDIR_CNTR}":
RUN mkdir -p "${ARG_APP_ANALYSIS_TEMPLATEDIR_CNTR}"

# mkdir "${ARG_APP_ANALYSIS_RESULTSDIR_CNTR}":
RUN mkdir -p "${ARG_APP_ANALYSIS_RESULTSDIR_CNTR}"

#? ---- . ---- ---- ---- ---- . ----
#? proper:
#? - update & prerequisites:
#? ---- . ---- ---- ---- ---- . ----

# update
RUN apt-get update

# install prerequisites:
RUN apt-get install -y\
    git\
    wget\
    curl\
    zip\
    unzip\
    gnupg

#? ---- . ---- ---- ---- ---- . ----
#? proper:
#? - python2 & python3:
#? ---- . ---- ---- ---- ---- . ----

# install python3
RUN apt-get install -y python${ARG_APP_ANALYSIS_PYTHON3_VERSION} python${ARG_APP_ANALYSIS_PYTHON3_VERSION}-venv

# create python3 venv:
RUN python${ARG_APP_ANALYSIS_PYTHON3_VERSION} -m venv "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}"
RUN rm -f "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin/python"
RUN rm -f "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin/pip"

# add python venv to $PATH:
# (permanemtly activates python venv for all users and all shells)
ENV PATH="${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin:$PATH"
RUN TEMP_CAT_ETC_ENVIROMENT="$(cat /etc/environment | sed "s|^PATH=\"\(.*\)\"$|PATH=\"${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin:\1\"|")"\
    [ ! -z "$TEMP_CAT_ETC_ENVIROMENT" ] && echo "$TEMP_CAT_ETC_ENVIROMENT" > /etc/environment\
    || echo "PATH=\"${PATH}\"" > /etc/environment

# test $PATH:
RUN echo "${PATH}" | grep "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin"
RUN cat /etc/environment | grep "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin"

# test if python, python3, python3.?? are setup corectly:
RUN which python || echo "python not found" | grep "python not found"
RUN which python3 | grep "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin/python3"
RUN which python${ARG_APP_ANALYSIS_PYTHON3_VERSION} | grep "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin/python${ARG_APP_ANALYSIS_PYTHON3_VERSION}"

# test if pip, pip3, pip3.?? are setup corectly:
RUN which pip || echo "pip not found" | grep "pip not found"
RUN which pip3 | grep "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin/pip3"
RUN which pip${ARG_APP_ANALYSIS_PYTHON3_VERSION} | grep "${ARG_APP_ANALYSIS_PYTHON3_VENV_CNTR}/bin/pip${ARG_APP_ANALYSIS_PYTHON3_VERSION}"

# install python3 packages with pip3
RUN pip3 install wheel
RUN pip3 install\
    numpy==2.1.2\
    scipy==1.14.1\
    matplotlib==3.9.2\
    wdata==0.2.0

# test python3 package versions
RUN pip3 list | grep "numpy *2.1.2"
RUN pip3 list | grep "scipy *1.14.1"
RUN pip3 list | grep "matplotlib *3.9.2"
RUN pip3 list | grep "wdata *0.2.0"

#? ---- . ---- ---- ---- ---- . ----
#? proper:
#? - node:
#? ---- . ---- ---- ---- ---- . ----

# install node:
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${ARG_APP_ANALYSIS_NODE_VERSION}.x nodistro main" > /etc/apt/sources.list.d/nodesource.list
RUN apt-get update
RUN apt-get install -y nodejs
# RUN npm install -g npm
# RUN npm install -g bun

# install yarn:
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | gpg --dearmor | tee /usr/share/keyrings/yarn.gpg >/dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/yarn.gpg] https://dl.yarnpkg.com/debian/ stable main" > /etc/apt/sources.list.d/yarn.list
RUN apt-get update
RUN apt-get install -y yarn

#? ---- . ---- ---- ---- ---- . ----
#? entrypoint script
#? ---- . ---- ---- ---- ---- . ----

# copy the entrypoint script
COPY "${ARG_APP_ANALYSIS_CONFIGDIR_HOST}/usr/local/bin/entrypoint" "/usr/local/bin/entrypoint"

# make the entrypoint script executable
RUN chmod +x "/usr/local/bin/entrypoint"

# env vars to be used by ENTRYPOINT & CMD:
ENV ENV_APP_ANALYSIS_WORKDIR="${ARG_APP_ANALYSIS_WORKDIR_CNTR}"
ENV ENV_APP_ANALYSIS_CONFIGDIR="${ARG_APP_ANALYSIS_CONFIGDIR_CNTR}"
ENV ENV_APP_ANALYSIS_TEMPLATEDIR="${ARG_APP_ANALYSIS_TEMPLATEDIR_CNTR}"
ENV ENV_APP_ANALYSIS_RESULTSDIR="${ARG_APP_ANALYSIS_RESULTSDIR_CNTR}"

# use the entrypoint script as the default command:
CMD /usr/local/bin/entrypoint\
    --app-analysis-workdir "${ENV_APP_ANALYSIS_WORKDIR}"\
    --app-analysis-configdir "${ENV_APP_ANALYSIS_CONFIGDIR}"\
    --app-analysis-templatedir "${ENV_APP_ANALYSIS_TEMPLATEDIR}"\
    --app-analysis-resultsdir "${ENV_APP_ANALYSIS_RESULTSDIR}"
