# Get Subject Data
wget --load-cookies /tmp/cookies.txt "https://doc.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://doc.google.com/uc?export=download&id=1oodtccHYvqT7If6nH-gipCDFQLNeWKz5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oodtccHYvqT7If6nH-gipCDFQLNeWKz5" -O dylan.tar.xz && rm -rf /tmp/cookies.txt
tar -xzvf dylan.tar.xz
rm dylan.tar.gz