#!/bin/bash

# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Unsupported OS"
    exit 1
fi

install_mqtt() {
    case "$OS" in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y mosquitto mosquitto-clients
            ;;
        centos|rhel)
            sudo yum install -y epel-release
            sudo yum install -y mosquitto mosquitto-clients
            ;;
        fedora)
            sudo dnf install -y mosquitto mosquitto-clients
            ;;
        arch)
            sudo pacman -S mosquitto
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
    echo "installed MQTT successfully."
}

install_mqtt

##start and enable mosquitto
#sudo systemctl enable --now mosquitto
#sudo systemctl start mosquitto
#systemctl status mosquitto

##test publish/subscribe on MQTT topic
#mosquitto_sub -h localhost -t "test/topic" &
#mosquitto_pub -h localhost -t "test/topic" -m "Hello MQTT"

##edit the following file to enable remote connections by adding listener 1883, allow_anonymous true
#sudo nano /etc/mosquitto/mosquitto.conf
##restart the service
#sudo systemctl restart mosquitto