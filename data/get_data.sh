# Get Subject Data; modify tarfile and link to point to all subject directories and not just dylan
wget --load-cookies /tmp/cookies.txt "https://doc.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://doc.google.com/uc?export=download&id=1mLjt7PXSQMk46-RclAR2MG_XCjLRu4vT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mLjt7PXSQMk46-RclAR2MG_XCjLRu4vT" -O dylan.tar.xz && rm -rf /tmp/cookies.txt
tar -xf dylan.tar.xz
rm dylan.tar.xz