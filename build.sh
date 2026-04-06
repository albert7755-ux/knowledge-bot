#!/bin/bash
# 安裝 LibreOffice（用於 PPT/Word 轉 PDF）
apt-get update -qq
apt-get install -y -qq libreoffice

# 安裝 Python 套件
pip install -r requirements.txt
