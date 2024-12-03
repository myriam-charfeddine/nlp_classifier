import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'naruto'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    def parse(self, response):
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():
            extracted_data = scrapy.Request ("https://naruto.fandom.com" + href, 
                                    callback=self.parse_jutsu)
            yield extracted_data

        for next_page in response.css('a.mw-nextlink'):
            yield response.follow(next_page, self.parse)

    def parse_jutsu(self, response):
        jutsu_name = response.css('span.mw-page-title-main::text').extract()[0]
        jutsu_name = jutsu_name.strip()

        div_html = response.css('div.mw-parser-output')[0].extract()

        soup = BeautifulSoup(div_html)
        # soup = BeautifulSoup(div_html).find('div')
        
        jutsu_type = ""
        if soup.find('aside'):
           aside_content = soup.find('aside')

           for cell in aside_content.find_all('div', {'class' : 'pi-data'}):
               if cell.find('h3'):
                   if cell.find('h3').text.strip()== 'Classification':
                       jutsu_type = cell.find('div').text.strip()

        soup.find('aside').decompose()  #remove 'aside' from soup #remove an element and free up memory associated with it

        jutsu_description = soup.text.strip()
        jutsu_description = jutsu_description.split('Trivia')[0].strip()

        return dict (
            jutsu_name = jutsu_name,
            jutsu_type = jutsu_type,
            jutsu_description = jutsu_description
        )
                
