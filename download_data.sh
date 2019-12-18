mkdir raw_data

wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone01.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone02.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone03.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone05.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone06.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone07.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone08.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone09.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone10.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone11.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone14.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone15.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone16.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone17.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone18.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone19.zip
wget https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone20.zip

unzip '*.zip'
rm -r *.zip
mv AIS_ASCII_by_UTM_Month/2017_v2/*.csv raw_data/
rm -r AIS_ASCII_by_UTM_Month


