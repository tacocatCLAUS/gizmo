from ScrapeSearchEngine.ScrapeSearchEngine import Duckduckgo

userAgent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15') #search on google "my user agent"
search = ('newest iphone news')  #Enter Anything for Search
duckduckgo = Duckduckgo(search, userAgent)

duckduckgoText = Duckduckgo(search, userAgent)

print(duckduckgoText)
