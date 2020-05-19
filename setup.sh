mkdir -p ~/.streamlit/

# echo "Downloading boost."
# mkdir Boost
# cd Boost
# wget https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.zip
# unzip boost_1_63_0.zip

# echo "Building starspace"
# cd ../starspace
# make

# echo "Building python wrapper"
# cd python
# chmod +x build.sh
# ./build.sh

# echo "Placing wrapper in app root"
# mv build/starwrap.so ../../..

echo "\
[general]\n\
email = \"diana@talentbait.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml