#!/usr/bin/env bash

rel_SCRIPTPATH=$( dirname -- ${BASH_SOURCE[0]}; );
source $rel_SCRIPTPATH/utils.sh

SCRIPTPATH=$(abspath $rel_SCRIPTPATH)/..;
chmod -R 777 $SCRIPTPATH;

#conda env create -f $SCRIPTPATH/environment.yml

### Add ZarrSeg directory to the PATH.

if ! echo $PATH | tr ":" "\n" | grep "ZarrSeg" &> /dev/null;
then
	echo "export PATH="$SCRIPTPATH:$PATH"" >> $HOME/.bashrc;
fi;

source $HOME/.bashrc;




