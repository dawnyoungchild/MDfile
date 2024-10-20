## 第1章 初识Scrapy

### 1.3 编写第一个Scrapy爬虫

#### 1.3.1 项目需求

从http://books.toscrape.com爬取书籍的书名和价格信息。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101514235320241015142353.png" alt="image-20241015142346094" style="zoom:80%;" />

#### 1.3.3 分析页面

##### 1.3.3.1 数据信息

**选取任意一本书**右键点击审查元素（检查），从HTML源代码中可以看到这本书的信息

![image-20241015142632436](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101514263220241015142632.png)

![image-20241015142911186](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101514291120241015142911.png)

每一本书的信息包裹在<article class="product_pod">元素中：书名信息在其下h3 > a元素的title属性中，如<a href="catalogue/a-light-in-the-attic_1000/index.html"title="A Light in the Attic">A Light in the ...</a>；书价信息在其下<p class="price_color">元素的文本中，如<p class="price_color">£51.77</p>。

##### 1.3.3.2 链接信息

![image-20241015143121233](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101514312120241015143121.png)

可以发现，下一页的URL在ul.pager > li.next > a元素的href属性中，是一个相对URL地址，如<li class="next"><a href="catalogue/page-2.html">next</a></li>。

#### 1.3.4 实现Spider

~~~python
# book_spider.py
import scrapy

class BooksSpider(scrapy.Spider):
    # 每一个爬虫的唯一标识
    name = "books"

    # 定义爬虫爬取的起始点，起始点可以是多个，这里只有一个
    start_urls = ['http://books.toscrape.com/']

    def parse(self, response):
        # 提取数据
        # 每一本书的信息在<article class="product_pod">中，我们使用
        # css()方法找到所有这样的article元素，并依次迭代
        for book in response.css('article.product_pod'):
            # 书名信息在article>h3>a元素的title属性里
            # 例如：<a title="A Light in the Attic">A Light in the ...</a>
            name = book.xpath('./h3/a/@title').extract_first()

            # 书价信息在 <p class="price_color">的TEXT中。
            # 例如：<p class="price_color">£51.77</p>
            price = book.css('p.price_color::text').extract_first()
            yield {
                'name': name,
                'price': price,
            }

            # 提取链接
            # 下一页的url在ul.pager>li.next>a里面
            # 例如：<li class="next"><a href="catalogue/page-2.html">next</a></li>
            next_url = response.css('ul.pager li.next a::attr(href)').extract_first()
            if next_url:
                # 如果找到下一页的URL，得到绝对路径，构造新的Request对象
                next_url = response.urljoin(next_url)
                yield scrapy.Request(next_url, callback=self.parse)
~~~

对上述代码的简要说明

- **name属性**

  一个Scrapy项目中可能有多个爬虫，每个爬虫的name属性是其自身的唯一标识，在一个项目中不能有同名的爬虫，本例中的爬虫取名为’books'。

- **start_urls属性**

  一个爬虫总要从某个（或某些）页面开始爬取，我们称这样的页面为起始爬取点，start_urls属性用来设置一个爬虫的起始爬取点。在本例中只有一个起始爬取点'http://books.toscrape.com'。

- **parse方法**

  当一个页面下载完成后，Scrapy引擎会回调一个我们指定的页面解析函数（默认为parse方法）解析页面。一个页面解析函数通常需要完成以下两个任务：

  ➢ 提取页面中的数据（使用XPath或CSS选择器）。

  ➢ 提取页面中的链接，并产生对链接页面的下载请求。

  页面解析函数通常被实现成一个**生成器函数**，每一项从页面中提取的数据以及每一个对链接页面的下载请求都**由yield语句提交给Scrapy引擎**。

#### 1.3.5 运行爬虫

在shell中执行scrapy crawl <SPIDER_NAME>命令运行爬虫’books'，并将爬取的数据存储到csv文件中

![image-20241015153018300](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101515301820241015153018.png)

 <SPIDER_NAME>是代码中的name属性

![image-20241015153144762](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101515314420241015153144.png)

查看文件内保存的内容

![image-20241015153413685](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101515341320241015153413.png)

------

## 第2章 编写Spider

### 2.1 Scrapy框架结构及工作原理

![image-20241015155256938](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101515525720241015155257.png)

**Scrapy框架中的各个组件**

| 组件          | 描述                                                 | 类型     |
| ------------- | ---------------------------------------------------- | -------- |
| ENGINE        | 引擎，框架的核心，其他所有组件在其控制下协同工作     | 内部组件 |
| SCHEDULER     | 调度器，负责对SPIDER提交的下载请求进行调度           | 内部组件 |
| DOWNLOADER    | 下载器，负责下载页面（发送HTTP请求/接收HTTP响应）    | 内部组件 |
| SPIDER        | 爬虫，负责提取页面中的数据，并产生对新页面的下载请求 | 用户实现 |
| MIDDLEWARE    | 中间件，负责对Request对象和Response对象进行处理      | 可选组件 |
| ITEM PIPELINE | 数据管道，负责对爬取到的数据进行处理                 | 可选组件 |

**Scrapy框架中的数据流**

| 对象     | 描述                   |
| -------- | ---------------------- |
| REQUEST  | Scrapy中的HTTP请求对象 |
| RESPONSE | Scrapy中的HTTP响应对象 |
| ITEM     | 从页面中爬取的一项数据 |

Request和Response是HTTP协议中的术语，即HTTP请求和HTTP响应，**Scrapy框架中定义了相应的Request和Response类**，这里的Item代表Spider从页面中爬取的一项数据。

**以上几种对象在框架中的流动过程**

● 当SPIDER要爬取某URL地址的页面时，需使用该URL构造一个Request对象，提交给ENGINE（图中的1）。

● Request对象随后进入SCHEDULER按某种算法进行排队，之后的某个时刻SCHEDULER将其出队，送往DOWNLOADER（图中的2、3、4）。

● DOWNLOADER根据Request对象中的URL地址发送一次HTTP请求到网站服务器，之后用服务器返回的HTTP响应构造出一个Response对象，其中包含页面的HTML文本（图中的5）。

● Response对象最终会被递送给SPIDER的页面解析函数（构造Request对象时指定）进行处理，页面解析函数从页面中提取数据，封装成Item后提交给ENGINE,Item之后被送往ITEM PIPELINES进行处理，最终可能由EXPORTER（图中没有显示）以某种数据格式写入文件（csv, json）；另一方面，页面解析函数还从页面中提取链接（URL），构造出新的Request对象提交给ENGINE（图中的6、7、8）。

### 2.2 Request和Response对象

#### 2.2.1 Request对象

~~~python
Request(url[, callback, method='GET', headers, body, cookies, meta,
            encoding='utf-8', priority=0, dont_filter=False, errback])
~~~

- **url（必选）**

  请求页面的url地址，bytes或str类型，如’http://www.python.org/doc'。

- **callback**

  页面解析函数，Callable类型，Request对象请求的页面下载完成后，由该参数指定的页面解析函数被调用。如果未传递该参数，默认调用Spider的parse方法。

- **method**

  HTTP请求的方法，默认为’GET'。

- **headers**

  HTTP请求的头部字典，**dict类型**，例如{'Accept': 'text/html', 'User-Agent':Mozilla/5.0'}。如果其中某项的值为None，就表示不发送该项HTTP头部，例如{'Cookie': None}，禁止发送Cookie。

- **body**

  HTTP请求的正文，bytes或str类型。

- **cookies**

  Cookie信息字典，**dict类型**，例如{'currency': 'USD', 'country': 'UY'}。

- **meta**

  Request的元数据字典，**dict类型**，用于给框架中其他组件传递信息，比如中间件ItemPipeline。其他组件可以使用Request对象的meta属性访问该元数据字典（request.meta），也用于给响应处理函数传递信息，详见Response的meta属性。

- **encoding**

  url和body参数的编码默认为’utf-8'。如果传入的url或body参数是str类型，就使用该参数进行编码。

- **priority**

  请求的优先级默认值为0，优先级高的请求优先下载。

- **dont_filter**

  默认情况下（dont_filter=False），对同一个url地址多次提交下载请求，后面的请求会被去重过滤器过滤（避免重复下载）。如果将该参数置为True，可以使请求避免被过滤，强制下载。例如，**在多次爬取一个内容随时间而变化的页面时（每次使用相同的url），可以将该参数置为True**。

- **errback**

  请求出现异常或者出现HTTP错误时（如404页面不存在）的回调函数。

#### 2.2.2 Response对象

Response对象用来描述一个HTTP响应，Response只是一个基类，根据响应内容的不同有如下子类：

● TextResponse

● HtmlResponse

● XmlResponse

**HtmlResponse对象的属性及方法**

- url

  HTTP响应的url地址，str类型

- status

  HTTP响应的状态码，int类型，例如200，404

- headers

  HTTP响应的头部，类字典类型，可以调用get或getlist方法对其访问，

  例如：

  ~~~python
  response.headers.get('Content-Type')
  response.headers.getlist('Set-Cookie')
  ~~~

- body

  HTTP响应正文，bytes类型

- text

  文本形式的HTTP响应正文，str类型，它是由response.body使用response.encoding解码得到的，即

  ~~~python
  reponse.text = response.body.decode(response.encoding)
  ~~~

- encoding

  HTTP响应正文的编码，它的值可能是从HTTP响应头部或正文中解析出来的

- request

  产生该HTTP响应的Request对象

- meta

  即response.request.meta，在构造Request对象时，可将要传递给响应处理函数的信息通过meta参数传入；响应处理函数处理响应时，通过response.meta将信息取出

- selector

  Selector对象用于在Response中提取数据。

- xpath（query）

  使用XPath选择器在Response中提取数据，实际上它是response.selector.xpath方法的快捷方式

- css（query）

  使用CSS选择器在Response中提取数据，实际上它是response.selector.css方法的快捷方式

- urljoin（url）

  用于构造绝对url。当传入的url参数是一个相对地址时，根据response.url计算出相应的绝对url。例如，response.url为http://www.example.com/a, url为b/index.html，调用response.urljoin(url)的结果为http://www.example.com/a/b/index.html。

其中常用的是以下3个方法：

● xpath(query)

● css(query)

● urljoin(url)

### 2.3 Spider开发流程

~~~python
# book_spider.py
import scrapy

class BooksSpider(scrapy.Spider):
    # 每一个爬虫的唯一标识
    name = "books"

    # 定义爬虫爬取的起始点，起始点可以是多个，这里是一个
    start_urls = ['http://books.toscrape.com/']

    def parse(self, response):
        # 提取数据
        # 每一本书的信息是在<article class="product_pod">中，使用css()方法找到所有这样的article元素，并依次迭代
        for book in response.css('article.product_pod'):
            # 书名信息在article>h3>a元素的title属性里
            # 例如：<a title="A Light in the Attic">A Light in the ...</a>
            name = book.xpath('./h3/a/@title').extract_first()

            # 书价信息在 <p class="price_color">的TEXT中。
            # 例如：<p class="price_color">£51.77</p>
            price = book.css('p.price_color::text').extract_first()
            yield {
                'name': name,
                'price': price,
            }

            # 提取链接
            # 下一页的url在ul.pager>li.next>a里面
            # 例如：<li class="next"><a href="catalogue/page-2.html">next</a></li>
            next_url = response.css('ul.pager li.next a::attr(href)').extract_first()
            if next_url:
                # 如果找到下一页的url，得到绝对路径，构造新的Request对象
                next_url = response.urljoin(next_url)
                yield scrapy.Request(next_url, callback=self.parse)
~~~

**实现一个Spider只需要完成下面4个步骤：**

**步骤01 继承scrapy.Spider。**

**步骤02 为Spider取名。**

**步骤03 设定起始爬取点。**

**步骤04 实现页面解析函数。**

#### 2.3.1 继承scrapy.Spider

~~~python
import scrapy
class BooksSpider(scrapy.Spider):
    ...
~~~

这个Spider基类实现了以下内容：

- 供Scrapy引擎调用的接口，例如用来创建Spider实例的类方法from_crawler。
- 供用户使用的实用工具函数，例如可以调用log方法将调试信息输出到日志。
- 供用户访问的属性，例如可以通过settings属性访问配置文件中的配置。

#### 2.3.2 为Spider命名

~~~python
# 在一个Scrapy项目中可以实现多个Spider，每个Spider需要有一个能够区分彼此的唯一标识，Spider的类属性name便是这个唯一标识。
class BooksSpider(scrapy.Spider):
    name = "books"
    ...
~~~

#### 2.3.3 设定起始爬取点

~~~python
class BooksSpider(scrapy.Spider):
    ...
    start_urls = ['http://books.toscrape.com/']
    ...
~~~

继承的Scrapy.Spider源代码：

~~~python
class Spider(object_ref):
    ...
    def start_requests(self):
        for url in self.start_urls:
            yield self.make_requests_from_url(url)

            def make_requests_from_url(self, url):
                return Request(url, dont_filter=True)

            def parse(self, response):
                raise NotImplementedError
    ...
~~~

从代码中可以看出，Spider基类的start_requests方法帮助我们构造并提交了Request对象。

由于起始爬取点的下载请求是由引擎调用Spider对象的start_requests方法产生的，因此我们也可以**在BooksSpider中实现start_requests方法（覆盖基类Spider的start_requests方法）**，直接构造并提交起始爬取点的Request对象。在某些场景下使用这种方式更加灵活，例如有时想为Request添加特定的HTTP请求头部，或想为Request指定特定的页面解析函数。

~~~python
# 自定义start_requests
class BooksSpider(scrapy.Spider):

    # start_urls = ['http://books.toscrape.com/']

    # 实现start_requests方法，替代start_urls类属性
    def start_requests(self):
        yield scrapy.Request('http://books.toscrape.com/',
                             callback=self.parse_book,
                             headers={'User-Agent': 'Mozilla/5.0'},
                             dont_filter=True)
        # 改用parse_book作为回调函数
    def parse_book(response):
        ...
~~~

#### 2.3.4 实现页面解析函数

页面解析函数也就是**构造Request对象时通过callback参数指定的回调函数（或默认的parse方法）**。页面解析函数是实现Spider中最核心的部分，它需要完成以下两项工作：

- 使用选择器提取页面中的数据，将数据封装后（Item或字典）提交给Scrapy引擎。
- 使用选择器或LinkExtractor提取页面中的链接，用其构造新的Request对象并提交给Scrapy引擎（下载链接页面）。

------

## 第3章 使用Selector提取数据

### 3.1 Selector对象

从页面中提取数据的核心技术是HTTP文本解析，在Python中常用以下模块处理此类问题：

- **BeautifulSoup**

  优点：API简单易用

  缺点：解析速度慢

- **lxml**

  优点：解析速度快

  缺点：API相对复杂

Scrapy综合上述两者优点实现了Selector类，它是基于lxml库构建的，并简化了API接口。在Scrapy中使用Selector对象提取页面中的数据，使用时先**通过XPath或CSS选择器选中页面中要提取的数据**，然后进行提取。

#### 3.1.1 创建对象

Selector类的实现位于**scrapy.selector**模块，创建Selector对象时，可将页面的HTML文档字符串传递给Selector构造器方法的**text参数**：

~~~xml
>>> from scrapy.selector import Selector
>>> text = '''
... <html>
    ...    <body>
    ...        <h1>Hello World</h1>
    ...        <h1>Hello Scrapy</h1>
    ...        <b>Hello python</b>
    ...        <ul>
    ...           <li>C++</li>
    ...           <li>Java</li>
    ...           <li>Python</li>
    ...        </ul>
    ...    </body>
    ... </html>
... '''
...
>>> selector = Selector(text=text)
<Selector xpath=None data='<html>\n       <body>\n          <h1>He'>
~~~

也可以使用一个Response对象构造Selector对象，将其传递给Selector构造器方法的response参数：

~~~xml
>>> from scrapy.selector import Selector
>>> from scrapy.http import HtmlResponse
>>> body = '''
... <html>
    ...    <body>
    ...        <h1>Hello World</h1>
    ...        <h1>Hello Scrapy</h1>
    ...        <b>Hello python</b>
    ...        <ul>
    ...           <li>C++</li>
    ...           <li>Java</li>
    ...           <li>Python</li>
    ...        </ul>
    ...    </body>
    ... </html>
... '''
...
>>> response = HtmlResponse(url='http://www.example.com', body=body, encoding='utf8')
>>> selector = Selector(response=response)
>>> selector
<Selector xpath=None data='<html>\n       <body>\n          <h1>He'>
~~~

#### 3.1.2 选中数据

调用Selector对象的xpath方法或css方法（传入XPath或CSS选择器表达式），可以选中文档中的某个或某些部分：

~~~xml
>>>selector_list=selector.xpath('//h1')           # 选中文档中所有的<h1>
>>>selector_list                           # 其中包含两个<h1>对应的Selector对象
    [<Selector query='//h1' data='<h1>Hello World</h1>'>,
    <Selector query='//h1' data='<h1>Hello Scrapy</h1>'>]
~~~

xpath和css方法返回一个SelectorList对象，其中包含每个被选中部分对应的Selector对象，SelectorList支持列表接口，可使用for语句迭代访问其中的每一个Selector对象：

~~~python
>>> for sel in selector_list:
    ...    print(sel.xpath('./text()'))
    ...
    [<Selector xpath='./text()' data='Hello World'>]
    [<Selector xpath='./text()' data='Hello Scrapy'>]
    ...    print(sel)
    ...
    <h1>Hello World</h1>
	<h1>Hello Scrapy</h1>
~~~

SelectorList对象也有xpath和css方法，调用它们的行为是：以接收到的参数分别调用其中每一个Selector对象的xpath或css方法，并将所有结果收集到一个新的SelectorList对象返回给用户：

~~~python
>>> selector_list.xpath('./text()')
[<Selector query='./text()' data='Hello World'>,
 <Selector query='./text()' data='Hello Scrapy'>]
>>> selector.xpath('.//ul').css('li').xpath('./text()')
[<Selector query='./text()' data='C++'>,
 <Selector query='./text()' data='Java'>,
 <Selector query='./text()' data='Python'>]
~~~

#### 3.1.3 提取数据

调用Selector或SelectorLis对象的以下方法可将选中的内容提取：

- **extract()**
- **re()**
- **extract_first()**（SelectorList专有）

- **re_first()**（SelectorList专有）

~~~python
# extract()方法
>>> sl = selector.xpath('.//li')
>>> sl
[<Selector query='.//li' data='<li>C++</li>'>,
 <Selector query='.//li' data='<li>Java</li>'>,
 <Selector query='.//li' data='<li>Python</li>'>]
>>> sl[0].extract()
'<li>C++</li>'
>>> sl = selector.xpath('.//li/text()')
>>> sl
[<Selector query='.//li/text()' data='C++'>,
 <Selector query='.//li/text()' data='Java'>,
 <Selector query='.//li/text()' data='Python'>]
>>> sl[1].extract()
'Java'
# 与SelectorList对象的xpath和css方法类似，SelectorList对象的extract方法内部会调用其中每个Selector对象的extract方法，并把所有结果收集到一个列表返回给用户：
>>> sl = selector.xpath('.//li/text()')
>>> sl
[<Selector query='.//li/text()' data='C++'>,
 <Selector query='.//li/text()' data='Java'>,
 <Selector query='.//li/text()' data='Python'>]
>>> s1.extract()
['C++', 'Java', 'Python']
~~~

~~~python
# extract_first()方法
# SelectorList专有,该方法返回其中第一个Selector对象调用extract方法的结果。通常，在SelectorList对象中只包含一个Selector对象时调用该方法，直接提取出Unicode字符串而不是列表
>>> s1 = selector.xpath('.//h1')
>>> s1
[<Selector query='.//h1' data='<h1>Hello World</h1>'>, <Selector query='.//h1' data='<h1>Hello Scrapy</h1>'>]
>>> s1.extract_first()
'<h1>Hello World</h1>'
~~~

~~~python
# re()方法，主要用于提取选中内容的某部分
>>> text2 = '''
 <ul>
    <li>Python学习手册 <b>价格： 99.00元</b></li>
    <li>Python核心编程 <b>价格： 88.00元</b></li>
    <li>Python基础教程 <b>价格： 80.00元</b></li>
 </ul>
 '''
>>> selector = Selector(text=text2)
>>> selector.xpath(',//li/b/text()')
>>> selector.xpath('.//li/b/text()')
[<Selector query='.//li/b/text()' data='价格： 99.00元'>,
 <Selector query='.//li/b/text()' data='价格： 88.00元'>,
 <Selector query='.//li/b/text()' data='价格： 80.00元'>]
>>> selector.xpath('.//li/b/text()').extract()
['价格： 99.00元', '价格： 88.00元', '价格： 80.00元']
>>> selector.xpath('.//li/b/text()').re('\d+.\d+')
['99.00', '88.00', '80.00']
# re_first方法调用re()
>>> selector.xpath('.//li/b/text()').re_first('\d+\.\d+')
'99.00'
~~~

### 3.2 Response内置Selector

~~~python
# 可以直接使用Response对象内置的Seletor对象
>>> from scrapy.http import HtmlResponse
>>> body= '''
 <html>
    <body>
        <h1>Hello World</h1>
        <h1>Hello Scrapy</h1>
        <b>Hello python</b>
        <ul>
            <li>C++</li>
            <li>Java</li>
            <li>Python</li>
        </ul>
    </body>
 </html>
'''
>>> response = HtmlResponse(url='http://www.example.com', body=body, encoding='utf-8')
>>> response.selector
<Selector query=None data='<html>\n    <body>\n        <h1>Hello W...'>
~~~

~~~python
# Response对象还提供了xpath和css方法，它们在内部分别调用内置Selector对象的xpath和css方法
>>> response.xpath('.//h1/text()').extract()
['Hello World', 'Hello Scrapy']
>>> response.css('li::text').extract()
['C++', 'Java', 'Python']
~~~

### 3.3 XPath

XPath即XML路径语言（XML Path Language），它是一种用来确定xml文档中某部分位置的语言。

**常用的xml文档节点类型**

- **根节点**：整个文档树的根
- **元素节点**：html、body、div、p、a
- **属性节点**：href
- **文本节点**：Hello world、Click here

**节点间的关系**

- **父子**：body是html的子节点，p和a是div的子节点。反过来，div是p和a的父节点。
- **兄弟**：p和a为兄弟节点
- **祖先/后裔**：body、div、p、a都是html的后裔节点；反过来html是body、div、p、a的祖先节点

#### 3.3.1 基础语法

| 表   达   式 | 描     述                                          |
| ------------ | -------------------------------------------------- |
| /            | 选中文档的根（root）                               |
| .            | 选中当前节点                                       |
| ..           | 选中当前节点的父节点                               |
| ELEMENT      | 选中子节点中所有ELEMENT元素节点                    |
| //ELEMENT    | 选中后代节点中所有ELEMENT元素节点                  |
| *            | 选中所有元素子节点                                 |
| text()       | 选中所有文本子节点                                 |
| @ATTR        | 选中名为ATTR的属性节点                             |
| @*           | 选中所有属性节点                                   |
| [谓语]       | 谓语用来查找某个特定的节点或者包含某个特定值的节点 |

**xpath使用举例**

~~~python
>>> from scrapy.selector import Selector
>>> from scrapy.http import HtmlResponse
>>> body = '''
<html>
    <head>
        <base href='http://example.com/'/>
        <title>Example website</title>
    </head>
    <body>
        <div id='images'>
           <a href='image1.html'>Name: Image 1<br/><img src='image1.jpg'/></a>
           <a href='image2.html'>Name: Image 2<br/><img src='image2.jpg'/></a>
           <a href='image3.html'>Name: Image 3<br/><img src='image3.jpg'/></a>
           <a href='image4.html'>Name: Image 4<br/><img src='image4.jpg'/></a>
           <a href='image5.html'>Name: Image 5<br/><img src='image5.jpg'/></a>
        </div>
    </body>
 </html>
 '''
>>> response = HtmlResponse(url='http://www.example.com', body=body, encoding='utf-8')

# /:描述一个从根开始的绝对路径
>>> response.xpath('/html')
[<Selector query='/html' data='<html>\n    <head>\n        <base href=...'>]
>>> response.xpath('/html/head')
[<Selector query='/html/head' data='<head>\n        <base href="http://exa...'>]

# E1/E2:选中E1子节点中的所有E2
# 选中div子节点中的所有a
>>> response.xpath('/html/body/div/a')
[<Selector query='/html/body/div/a' data='<a href="image1.html">Name: Image 1<b...'>,
 <Selector query='/html/body/div/a' data='<a href="image2.html">Name: Image 2<b...'>,
 <Selector query='/html/body/div/a' data='<a href="image3.html">Name: Image 3<b...'>,
 <Selector query='/html/body/div/a' data='<a href="image4.html">Name: Image 4<b...'>,
 <Selector query='/html/body/div/a' data='<a href="image5.html">Name: Image 5<b...'>]

# //E:选中文档中的所有E，无论在什么位置
# 选中文档中的所有a
>>> response.xpath('//a')
[<Selector query='//a' data='<a href="image1.html">Name: Image 1<b...'>,
 <Selector query='//a' data='<a href="image2.html">Name: Image 2<b...'>,
 <Selector query='//a' data='<a href="image3.html">Name: Image 3<b...'>,
 <Selector query='//a' data='<a href="image4.html">Name: Image 4<b...'>,
 <Selector query='//a' data='<a href="image5.html">Name: Image 5<b...'>]

# E1//E2：选中E1后代节点中的所有E2，无论在后代中的什么位置
# 选中body后代中的所有img
>>> response.xpath('/html/body//img')
[<Selector query='/html/body//img' data='<img src="image1.jpg">'>,
 <Selector query='/html/body//img' data='<img src="image2.jpg">'>,
 <Selector query='/html/body//img' data='<img src="image3.jpg">'>,
 <Selector query='/html/body//img' data='<img src="image4.jpg">'>,
 <Selector query='/html/body//img' data='<img src="image5.jpg">'>]

# E/text()：选中E的文本子节点
# 选中所有a的文本
>>> sel = response.xpath('//a/text()')
>>> sel
[<Selector query='//a/text()' data='Name: Image 1'>,
 <Selector query='//a/text()' data='Name: Image 2'>,
 <Selector query='//a/text()' data='Name: Image 3'>,
 <Selector query='//a/text()' data='Name: Image 4'>,
 <Selector query='//a/text()' data='Name: Image 5'>]
>>> sel.extract()
['Name: Image 1', 'Name: Image 2', 'Name: Image 3', 'Name: Image 4', 'Name: Image 5']

# E/*：选中E的所有元素子节点
# 选中html的所有元素子节点
>>> response.xpath('/html/*')
[<Selector query='/html/*' data='<head>\n        <base href="http://exa...'>, <Selector query='/html/*' data='<body>\n        <div id="images">\n    ...'>]
# 选中div的所有后代元素节点
>>> response.xpath('/html/body/div//*')
[<Selector query='/html/body/div//*' data='<a href="image1.html">Name: Image 1<b...'>,
 <Selector query='/html/body/div//*' data='<br>'>,
 <Selector query='/html/body/div//*' data='<img src="image1.jpg">'>,
 <Selector query='/html/body/div//*' data='<a href="image2.html">Name: Image 2<b...'>,
 <Selector query='/html/body/div//*' data='<br>'>,
 <Selector query='/html/body/div//*' data='<img src="image2.jpg">'>,
 <Selector query='/html/body/div//*' data='<a href="image3.html">Name: Image 3<b...'>,
 <Selector query='/html/body/div//*' data='<br>'>,
 <Selector query='/html/body/div//*' data='<img src="image3.jpg">'>,
 <Selector query='/html/body/div//*' data='<a href="image4.html">Name: Image 4<b...'>,
 <Selector query='/html/body/div//*' data='<br>'>,
 <Selector query='/html/body/div//*' data='<img src="image4.jpg">'>,
 <Selector query='/html/body/div//*' data='<a href="image5.html">Name: Image 5<b...'>,
 <Selector query='/html/body/div//*' data='<br>'>,
 <Selector query='/html/body/div//*' data='<img src="image5.jpg">'>]

# */E：选中孙节点中的所有E
# 选中div孙节点中的所有img
>>> response.xpath('//div/*/img')
[<Selector query='//div/*/img' data='<img src="image1.jpg">'>,
 <Selector query='//div/*/img' data='<img src="image2.jpg">'>,
 <Selector query='//div/*/img' data='<img src="image3.jpg">'>,
 <Selector query='//div/*/img' data='<img src="image4.jpg">'>,
 <Selector query='//div/*/img' data='<img src="image5.jpg">'>]

# E/@ATTR：选中E的ATTR属性
# 选中所有img的src属性
>>> response.xpath('//img/@src')
[<Selector query='//img/@src' data='image1.jpg'>,
 <Selector query='//img/@src' data='image2.jpg'>,
 <Selector query='//img/@src' data='image3.jpg'>,
 <Selector query='//img/@src' data='image4.jpg'>,
 <Selector query='//img/@src' data='image5.jpg'>]

# //@ATTR：选中文档中所有ATTR属性
# 选中所有的href属性
>>> response.xpath('//@href')
[<Selector query='//@href' data='http://example.com/'>,
 <Selector query='//@href' data='image1.html'>,
 <Selector query='//@href' data='image2.html'>,
 <Selector query='//@href' data='image3.html'>,
 <Selector query='//@href' data='image4.html'>,
 <Selector query='//@href' data='image5.html'>]

# E/@*：选中E的所有属性
# 获取第一个a下img的所有属性（这里只有src一个属性）
>>> response.xpath('//a[1]/img/@*')
[<Selector query='//a[1]/img/@*' data='image1.jpg'>]
# 获取所有a下img的所有属性
>>> response.xpath('//a/img/@*')
[<Selector query='//a/img/@*' data='image1.jpg'>,
 <Selector query='//a/img/@*' data='image2.jpg'>,
 <Selector query='//a/img/@*' data='image3.jpg'>,
 <Selector query='//a/img/@*' data='image4.jpg'>,
 <Selector query='//a/img/@*' data='image5.jpg'>]

# .：选中当前节点，用来描述相对路径
# 获取第1个a的选择器对象
>>> sel = response.xpath('//a')[0]
>>> sel
<Selector query='//a' data='<a href="image1.html">Name: Image 1<b...'>
# 不用.会获取到文档中所有的img
>>> sel.xpath('//img')
[<Selector query='//img' data='<img src="image1.jpg">'>,
 <Selector query='//img' data='<img src="image2.jpg">'>,
 <Selector query='//img' data='<img src="image3.jpg">'>,
 <Selector query='//img' data='<img src="image4.jpg">'>,
 <Selector query='//img' data='<img src="image5.jpg">'>]
# 使用.获取当前节点后代中的所有img
>>> sel.xpath('.//img')
[<Selector query='.//img' data='<img src="image1.jpg">'>]

# ..：选中当前节点的父节点，用来描述相对路径
# 选中所有img的父节点
>>> response.xpath('//img/..')
[<Selector query='//img/..' data='<a href="image1.html">Name: Image 1<b...'>,
 <Selector query='//img/..' data='<a href="image2.html">Name: Image 2<b...'>,
 <Selector query='//img/..' data='<a href="image3.html">Name: Image 3<b...'>,
 <Selector query='//img/..' data='<a href="image4.html">Name: Image 4<b...'>,
 <Selector query='//img/..' data='<a href="image5.html">Name: Image 5<b...'>]

# node[谓语]：谓语用来查找某个特定的节点或者包含某个特定值的节点
# 选中所有a中的第3个
>>> response.xpath('//a[3]')
[<Selector query='//a[3]' data='<a href="image3.html">Name: Image 3<b...'>]
# 使用last函数，选中最后一个
>>> response.xpath('//a[last()]')
[<Selector query='//a[last()]' data='<a href="image5.html">Name: Image 5<b...'>]
# 使用position函数，选中前3个
>>> response.xpath('//a[position()<=3]')
[<Selector query='//a[position()<=3]' data='<a href="image1.html">Name: Image 1<b...'>,
 <Selector query='//a[position()<=3]' data='<a href="image2.html">Name: Image 2<b...'>,
 <Selector query='//a[position()<=3]' data='<a href="image3.html">Name: Image 3<b...'>]
# 使用position函数，选中2-4个
>>> response.xpath('//a[position()<=4 and position()>=2]')
[<Selector query='//a[position()<=4 and position()>=2]' data='<a href="image2.html">Name: Image 2<b...'>,
 <Selector query='//a[position()<=4 and position()>=2]' data='<a href="image3.html">Name: Image 3<b...'>,
 <Selector query='//a[position()<=4 and position()>=2]' data='<a href="image4.html">Name: Image 4<b...'>]
# 选中所有含有id属性的div
>>> response.xpath('//div[@id]')
[<Selector query='//div[@id]' data='<div id="images">\n           <a href=...'>]
# 选中所有含有id属性且值为“images”的div
>>> response.xpath('//div[@id="images"]')
[<Selector query='//div[@id="images"]' data='<div id="images">\n           <a href=...'>]
~~~

#### 3.3.2 常用函数

~~~python
# string(arg)：返回参数的字符串值
>>> from scrapy.selector import Selector
>>> text = '<a href="#">Click here to go to the <strong>Next Page</strong></a>'
>>> sel = Selector(text=text)
>>> sel
<Selector query=None data='<html><body><a href="#">Click here to...'>
# 以下做法和sel.xpath('/html/body/a/strong/text()')得到结果相同
>>> sel.xpath('string(/html/body/a/strong)').extract()
['Next Page']
# 因为‘click here to go to the’和‘Next page’在不同元素下，想获取到完整字符需要使用string函数而不能用text函数
>>> sel.xpath('/html/body/a//text()').extract()
['Click here to go to the ', 'Next Page']
>>> sel.xpath('string(/html/body/a)').extract()
['Click here to go to the Next Page']
~~~

~~~python
# contains(str1, str2)：判断str1中是否包含str2，返回布尔值
text = '''
<div>
    <p class="small info">hello world</p>
    <p class="normal info">hello scrapy</p>
 </div>
 '''
>>> sel = Selector(text=text)
>>> sel.xpath('//p[contains(@class, "small")]')
# 选择class属性中包含‘small’的p元素
[<Selector query='//p[contains(@class, "small")]' data='<p class="small info">hello world</p>'>]
# 选择class属性中包含‘info’的p元素
>>> sel.xpath('//p[contains(@class, "info")]')
[<Selector query='//p[contains(@class, "info")]' data='<p class="small info">hello world</p>'>, <Selector query='//p[contains(@class, "info")]' data='<p class="normal info">hello scrapy</p>'>]
~~~

### 3.4 CSS选择器

**CSS选择器常用的基本语法**

| 表达式                                  | 描述                                                   | 例子                                    |
| --------------------------------------- | ------------------------------------------------------ | --------------------------------------- |
| *                                       | 选中所有元素                                           | *                                       |
| E                                       | 选中E元素                                              | p                                       |
| E1,E2                                   | 选中E1和E2元素                                         | div，pre                                |
| E1E2                                    | 选中E1后代元素中的E2元素                               | div p                                   |
| E1>E2                                   | 选中E1子元素中的E2元素                                 | div>p                                   |
| E1+E2                                   | 选中E1兄弟元素中的E2元素                               | p+strong                                |
| .CLASS                                  | 选中CLASS属性包含CLASS的元素                           | .info                                   |
| #ID                                     | 选中id属性为ID的元素                                   | #main                                   |
| [ATTR]                                  | 选中包含ATTR属性的元素                                 | [href]                                  |
| [ATTR=VALUE]                            | 选中包含ATTR属性且值为VALUE的元素                      | [method=post]                           |
| [ATTR~=VALUE]                           | 选中包含ATTR属性且值包含VALUE的元素                    | [class~=clearfix]                       |
| E:nth-child(n)<br />E:nth-last-child(n) | 选中E元素，且该元素必须是其父元素的（倒数）第n个子元素 | a:nth-child(1)<br />a:nth-last-child(2) |
| E:first-child<br />E:last-child         | 选中E元素，且该元素必须是其父元素的（倒数）第1个子元素 | a:first-child<br />a:last-child         |
| E:empty                                 | 选中没有子元素的E元素                                  | div:empty                               |
| E::text                                 | 选中E元素的文本节点（Text Node）                       | p::text                                 |

**CSS使用举例**

~~~python
# 创建一个HTML文档并构建HtmlResponse对象
>>> from scrapy.selector import Selector
>>> from scrapy.http import HtmlResponse
>>> body = '''
 <html>
    <head>
        <base href='http://example.com/'/>
        <title>Example website</title>
    </head>
    <body>
        <div id='images-1'style="width: 1230px; ">
           <a href='image1.html'>Name: Image 1<br/><img src='image1.jpg'/></a>
           <a href='image2.html'>Name: Image 2<br/><img src='image2.jpg'/></a>
           <a href='image3.html'>Name: Image 3<br/><img src='image3.jpg'/></a>
        </div>
        
		<br/><img src='image6.jpg'/>
		
        <div id='images-2'class='small'>
           <a href='image4.html'>Name: Image 4<br/><img src='image4.jpg'/></a>
           <a href='image5.html'>Name: Image 5<br/><img src='image5.jpg'/></a>
        </div>
    </body>
 </html>
 '''
>>> response = HtmlResponse(url='http://www.example', body=body, encoding='utf-8')
~~~

~~~python
# E:选中E元素
# 选中所有的img
>>> response.css('img')
[<Selector query='descendant-or-self::img' data='<img src="image1.jpg">'>,
 <Selector query='descendant-or-self::img' data='<img src="image2.jpg">'>,
 <Selector query='descendant-or-self::img' data='<img src="image3.jpg">'>,
 <Selector query='descendant-or-self::img' data='<img src="image6.jpg">'>,
 <Selector query='descendant-or-self::img' data='<img src="image4.jpg">'>,
 <Selector query='descendant-or-self::img' data='<img src="image5.jpg">'>]

# E1, E2：选中E1和E2元素
# 选中所有base和title
>>> response.css('base, title')
[<Selector query='descendant-or-self::base | descendant-or-self::title' data='<base href="http://example.com/">'>,
 <Selector query='descendant-or-self::base | descendant-or-self::title' data='<title>Example website</title>'>]

# E1 E2：选中E1后代元素中的E2元素
# 选中div后代中的img
>>> response.css('div img')
[<Selector query='descendant-or-self::div/descendant-or-self::*/img' data='<img src="image1.jpg">'>, <Selector query='descendant-or-self::div/descendant-or-self::*/img' data='<img src="image2.jpg">'>, <Selector query='descendant-or-self::div/descendant-or-self::*/img' data='<img src="image3.jpg">'>, <Selector query='descendant-or-self::div/descendant-or-self::*/img' data='<img src="image4.jpg">'>, <Selector query='descendant-or-self::div/descendant-or-self::*/img' data='<img src="image5.jpg">'>]

# E1>E2：选中E1子元素中的E2元素
# 选中body子元素中的div
>>> response.css('body>div')
[<Selector query='descendant-or-self::body/div' data='<div id="images-1" style="width: 1230...'>, <Selector query='descendant-or-self::body/div' data='<div id="images-2" class="small">\n   ...'>]

# [ATTR]：选中包含ATTR属性的元素
# 选中包含style属性的元素
>>> response.css('[style]')
[<Selector query='descendant-or-self::*[@style]' data='<div id="images-1" style="width: 1230...'>]

#  [ATTR=VALUE]：选中包含ATTR属性且值为VALUE的元素
# 选中属性id为images-1的元素
>>> response.css('[id=images-1]')
[<Selector query="descendant-or-self::*[@id = 'images-1']" data='<div id="images-1" style="width: 1230...'>]

# E:nth-child(n)：选中E元素，且该元素必须是其父元素的第n个子元素
# 选中每个div的第一个a
>>> response.css('div>a:nth-child(1)')
[<Selector query='descendant-or-self::div/a[count(preceding-sibling::*) = 0]' data='<a href="image1.html">Name: Image 1<b...'>,
 <Selector query='descendant-or-self::div/a[count(preceding-sibling::*) = 0]' data='<a href="image4.html">Name: Image 4<b...'>]
# 选中第1个div的第1个a
>>> response.css('div:nth-child(1)>a:nth-child(1)')
[<Selector query='descendant-or-self::div[count(preceding-sibling::*) = 0]/a[count(preceding-sibling::*) = 0]' data='<a href="image1.html">Name: Image 1<b...'>]

# E:first-child：选中E元素，该元素必须是其父元素的第一个子元素
# E:last-child：选中E元素，该元素必须是其父元素的倒数第一个子元素
# 选中第1个div的最后1个a
>>> response.css('div:first-child>a:last-child')
[<Selector query='descendant-or-self::div[count(preceding-sibling::*) = 0]/a[count(following-sibling::*) = 0]' data='<a href="image3.html">Name: Image 3<b...'>]

# E::text：选中E元素的文本节点
# 选中所有a的文本
>>> sel = response.css('a::text')
>>> sel
[<Selector query='descendant-or-self::a/text()' data='Name: Image 1'>,
 <Selector query='descendant-or-self::a/text()' data='Name: Image 2'>,
 <Selector query='descendant-or-self::a/text()' data='Name: Image 3'>,
 <Selector query='descendant-or-self::a/text()' data='Name: Image 4'>,
 <Selector query='descendant-or-self::a/text()' data='Name: Image 5'>]
>>> sel.extract()
['Name: Image 1', 'Name: Image 2', 'Name: Image 3', 'Name: Image 4', 'Name: Image 5']
~~~

------

##  第4章 使用Item封装数据

### 4.1 Item和Field

用户可以使用Scrapy提供的Item和Field两个类，对数据进行自定义封装

- **Item基类**

  自定义数据类（如BookItem）的基类

- **Field类**

  用来描述自定义数据类包含哪些字段（如name、price等）

自定义一个数据类，只需继承Item，并创建一系列Field对象的类属性即可。

~~~python
# 以定义书籍信息BookItem为例，它包含两个字段，分别为书的名字name和书的价格price，代码如下
>>> from scrapy import Item, Field
>>> class BookItem(Item):
	name = Field()
	price = Field()
    
# 创建BookItem对象的方式类似字典
>>> book1 = BookItem(name='Needful Things', price=45.0)
>>> book1
{'name': 'Needful Things', 'price': 45.0}
>>> book2 = BookItem()
>>> book2
{}
>>> book2['name'] = 'Life of Pi'
>>> book2['price'] = 32.5
>>> book2
{'name': 'Life of Pi', 'price': 32.5}

# 访问BookItem对象
>>> book = BookItem(name='Needful Things', price=45.0)
>>> book['name']
'Needful Things'
>>> book.get('price', 60.0)
45.0
>>> list(book.items())
[('name', 'Needful Things'), ('price', 45.0)]
~~~

改写第1章example项目中的代码，使用Item和Field定义BookItem类，用其封装爬取到的书籍信息项目目录下的items.py文件供用户实现各种自定义的数据类，在items.py中实现BookItem。

~~~python
# 定义BookItem类
from scrapy import Item, Field

class BookItem(Item):
    name = Field()
    price = Field()
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101710143420241017101434.png" alt="image-20241017101427487" style="zoom:80%;" />

~~~python
# 使用 BookItem代替字典

from ..items import BookItem

class BooksSpider(scrapy.Spider):
    ...
    def parse(self, response):
        for sel in response.css('article.product_pod'):
            book = BookItem()
            book['name'] = sel.xpath('./h3/a/@title').extract_first()
            book['price'] = sel.css('p.price_color::text').extract_first()
            yield book
    ...
~~~

### 4.2 拓展Item字类

有些时候，我们可能要根据需求对已有的自定义数据类（Item子类）进行拓展。例如，example项目中又添加了一个新的Spider，它负责在另外的图书网站爬取国外书籍（中文翻译版）的信息，此类书籍的信息比之前多了一个译者字段，此时可以继承BookItem定义一个ForeignBookItem类，在其中添加一个译者字段，代码如下：

~~~python
>>> class ForeignBookItem(BookItem):
        translator=Field()

>>> book = ForeignBookItem()
>>> book['name'] = ’巴黎圣母院’
>>> book['price'] = 20.0
>>> book['translator'] = ’陈敬容’
~~~

### 4.3 Field元数据

通过Field元数据对数据进行处理

~~~python
class ExampleItem(Item):
    x=Field(a='hello', b=[1, 2, 3])      #x有两个元数据，a是个字符串，b是个列表
    y=Field(a=lambda x: x**2)      #y有一个元数据，a是个函数
# 访问一个ExampleItem对象的Item属性，将得到一个包含所有Field对象的字典
>>> e = ExampleItem(x=100, y=200)
>>> e.fields
{'x': {'a': 'hello', 'b': [1, 2, 3]},
 'y': {'a': <function __main__.ExampleItem.<lambda>>}}
>>> type(e.fields['x'])
scrapy.item.Field
>>> type(e.fields['y'])
scrapy.item.Field
# 实际上，Field是Python字典的子类，可以通过键获取Field对象中的元数据
>>> issubclass(Field, dict)
True
>>>field_x=e.fields['x']        # 注意，不要混淆e.fields['x']和e['x']
>>> field_x
{'a': 'hello', 'b': [1, 2, 3]}
>>> field_x['a']
'hello'
>>> field_y = e.fields['y']
>>> field_y
{'a': <function __main__.ExampleItem.<lambda>>}
>>> field_y.get('a', lambda x: x)
<function __main__.ExampleItem.<lambda>>
~~~

应用举例

~~~python
# 对书籍作者格式进行处理，对多名作者使用“|”符号进行处理
>>> from scrapy import Item, Field
>>> class BookItem(Item):
	authors = Field(serializer=lambda x: '|'.join(x))
    
>>> book = BookItem()
>>> book['authors'] = ['李雷', '韩梅梅', '吉姆']
>>> book
{'authors': ['李雷', '韩梅梅', '吉姆']}
# 在写入csv文件时，作者格式变成“李雷|韩梅梅|吉姆”
~~~

------

## 第5章 使用Item Pipeline处理数据

在Scrapy中，Item Pipeline是处理数据的组件，一个Item Pipeline就是一个包含特定接口的类，通常只负责一种功能的数据处理，在一个项目中可以同时启用多个Item Pipeline，它们按指定次序级联起来，形成一条数据处理流水线。以下是Item Pipeline的几种典型应用：

-  清洗数据
-  验证数据的有效性
-  过滤掉重复的数据
- 将数据存入数据库

### 5.1 Item Pipeline

#### 5.1.1 实现Item Pipeline

**应用实例**

在第1章的example项目中，我们爬取到的书籍价格是以英镑为单位的：

~~~python
$ scrapy crawl books -o books.csv
...
$head-5 books.csv    # 查看文件开头的5行
name, price
A Light in the Attic, £51.77
Tipping the Velvet, £53.74
Soumission, £50.10
Sharp Objects, £47.82
~~~

使用item pipeline实现英镑到人民币的转换

**在创建一个Scrapy项目时，会自动生成一个pipelines.py文件，它用来放置用户自定义的Item Pipeline。**

~~~python
class PriceConverterPipeline(object):

    # 英镑兑换人民币汇率
    exchange_rate = 8.5309

    def process_item(self, item, spider):
        # 提取item的price字段（如£53.74）
        # 去掉前面英镑符号£，转换为float类型，乘以汇率
        price = float(item['price'][1:]) * self.exchange_rate

        # 保留2位小数，赋值回item的price字段
        item['price'] = '¥%.2f' % price

        return item
~~~

#### 5.1.2 启用Item Pipeline

启用某个（或某些）Item Pipeline，需要在配置文件settings.py中进行配置：

~~~python
# ITEM_PIPELINES是一个字典，我们把想要启用的Item Pipeline添加到这个字典中，其中每一项的键是每一个Item Pipeline类的导入路径，值是一个0~1000的数字，同时启用多个Item Pipeline时，Scrapy根据这些数值决定各Item Pipeline处理数据的先后次序，数值小的在前。
ITEM_PIPELINES = {
    'example.pipelines.PriceConverterPipeline': 300,
}
~~~

### 5.2 更多例子

#### 5.2.1 过滤重复数据

爬取的书名去重

~~~python
from scrapy.exceptions import DropItem

class DuplicatesPipeline(object):

    def __init__(self):

        self.book_set = set()

        def process_item(self, item, spider):
            name = item['name']
            if name in self.book_set:
                raise DropItem("Duplicate book found: %s" % item)

                self.book_set.add(name)
                return item
~~~

![image-20241018080925076](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101808093220241018080932.png)

~~~python
# settings.py配置文件中启用DuplicatesPipeline

ITEM_PIPELINES = {
    'example.pipelines.PriceConverterPipeline': 300,
    'example.pipelines.DuplicatesPipeline': 350,
}
~~~

![image-20241018081515881](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101808151520241018081515.png)

查看log信息可以找到重复项

![image-20241018081700514](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101808170020241018081700.png)

#### 5.2.2 将数据存入MongoDB

通过Item Pipeline将爬取到的数据存入数据库

~~~python
from scrapy.item import Item
import pymongo

class MongoDBPipeline(object):

    DB_URI = 'mongodb://localhost:27017/'
    DB_NAME = 'scrapy_data'

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.DB_URI)
        self.db = self.client[self.DB_NAME]

        def close_spider(self, spider):
            self.client.close()

            def process_item(self, item, spider):
                collection = self.db[spider.name]
                post = dict(item) if isinstance(item, Item) else item
                collection.insert_one(post)
                return item
~~~

对上述代码解释如下。

-  在类属性中定义两个常量：

  ➢ DB_URI数据库的URI地址

  ➢ DB_NAME数据库的名字

- 在Spider整个爬取过程中，数据库的连接和关闭操作只需要进行一次，应在开始处理数据之前连接数据库，并在处理完所有数据之后关闭数据库。因此实现以下两个方法（在Spider打开和关闭时被调用）：

  ➢ open_spider(spider)

  ➢ close_spider(spider)

  分别在open_spider和close_spider方法中实现数据库的连接与关闭。

- 在process_item中实现MongoDB数据库的写入操作，使用self.db和spider.name获取一个集合（collection），然后将数据插入该集合，集合对象的insert_one方法需传入一个字典对象（不能传入Item对象），因此在调用前先对item的类型进行判断，如果item是Item对象，就将其转换为字典。

~~~python
# settings.py配置文件中启用MongoDBPipeline
ITEM_PIPELINES = {
    'example.pipelines.PriceConverterPipeline': 300,
    'example.pipelines.DuplicatesPipeline': 350,
    'example.pipelines.MongoDBPipeline': 400,
}
~~~

运行并查看结果

~~~python
$ scrapy crawl books
...
$ mongo
MongoDB shell version: 2.4.9
    connecting to: test
        > use scrapy_data
        switched to db scrapy_data
        > db.books.count()
        1000
        > db.books.find()
        { "_id" : ObjectId("58ae39a89dcd191973cc588f"), "price" : "¥441.64", "name" : "A Light in the
         Attic" }
         { "_id" : ObjectId("58ae39a89dcd191973cc5890"), "price" : "¥458.45", "name" : "Tipping the
          Velvet" }
          { "_id" : ObjectId("58ae39a89dcd191973cc5891"), "price" : "¥427.40", "name" : "Soumission" }
          { "_id" : ObjectId("58ae39a89dcd191973cc5892"), "price" : "¥407.95", "name" : "Sharp Objects" }
          { "_id" : ObjectId("58ae39a89dcd191973cc5893"), "price" : "¥462.63", "name" : "Sapiens: A Brief
           History of Humankind" }
           { "_id" : ObjectId("58ae39a89dcd191973cc5894"), "price" : "¥193.22", "name" : "The Requiem
            Red" }
            { "_id" : ObjectId("58ae39a89dcd191973cc5895"), "price" : "¥284.42", "name" : "The Dirty Little
             Secrets of Getting Your Dream Job" }
             { "_id" : ObjectId("58ae39a89dcd191973cc5896"), "price" : "¥152.96", "name" : "The Coming
              Woman: A Novel Based on the Life of the Infamous Feminist, Victoria Woodhull" }
              { "_id" : ObjectId("58ae39a89dcd191973cc5897"), "price" : "¥192.80", "name" : "The Boys in the
               Boat: Nine Americans and Their Epic Quest for Gold at the 1936 Berlin Olympics" }
               { "_id" : ObjectId("58ae39a89dcd191973cc5898"), "price" : "¥444.89", "name" : "The Black Maria" }
               { "_id" : ObjectId("58ae39a89dcd191973cc5899"), "price" : "¥119.35", "name" : "Starving Hearts
                (Triangular Trade Trilogy, #1)" }
                 { "_id" : ObjectId("58ae39a89dcd191973cc589a"), "price" : "¥176.25", "name" : "Shakespeare's
                  Sonnets" }
                  { "_id" : ObjectId("58ae39a89dcd191973cc589b"), "price" : "¥148.95", "name" : "Set Me Free" }
                  { "_id" : ObjectId("58ae39a89dcd191973cc589c"), "price" : "¥446.08", "name" : "Scott Pilgrim's
                   Precious Little Life (Scott Pilgrim #1)" }
                                         { "_id" : ObjectId("58ae39a89dcd191973cc589d"), "price" : "¥298.75", "name" : "Rip it Up and Start
                                          Again" }
                                          { "_id" : ObjectId("58ae39a89dcd191973cc589e"), "price" : "¥488.39", "name" : "Our Band Could Be
                                           Your Life: Scenes from the American Indie Underground, 1981-1991" }
                                           { "_id" : ObjectId("58ae39a89dcd191973cc589f"), "price" : "¥203.72", "name" : "Olio" }
                                           { "_id" : ObjectId("58ae39a89dcd191973cc58a0"), "price" : "¥320.68", "name" : "Mesaerion: The
                                            Best Science Fiction Stories 1800-1849" }
                                            { "_id" : ObjectId("58ae39a89dcd191973cc58a1"), "price" : "¥437.89", "name" : "Libertarianism for
                                             Beginners" }
                                             { "_id" : ObjectId("58ae39a89dcd191973cc58a2"), "price" : "¥385.34", "name" : "It's Only the
                                              Himalayas" }
                                              Type "it" for more
~~~

在上述实现中，数据库的URI地址和数据库的名字硬编码在代码中，如果希望通过配置文件设置它们，只需稍作改动，代码如下：

~~~python
from scrapy.item import Item
import pymongo

class MongoDBPipeline(object):
    @classmethod
    def from_crawler(cls, crawler):
        cls.DB_URI = crawler.settings.get('MONGO_DB_URI',
                                          'mongodb://localhost:27017/')
        cls.DB_NAME = crawler.settings.get('MONGO_DB_NAME', 'scrapy_data')

        return cls()

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.DB_URI)
        self.db = self.client[self.DB_NAME]
        def close_spider(self, spider):
            self.client.close()

            def process_item(self, item, spider):
                collection = self.db[spider.name]
                post = dict(item) if isinstance(item, Item) else item
                collection.insert_one(post)

                return item
~~~

对上述改动解释如下：

-  增加类方法from_crawle（r cls, crawler），替代在类属性中定义DB_URI和DB_NAME。

- 如果一个Item Pipeline定义了from_crawler方法，Scrapy就会调用该方法来创建Item Pipeline对象。该方法有两个参数：

  ➢ cls Item Pipeline类的对象（这里为MongoDBPipeline类对象）

  ➢ crawler Crawler是Scrapy中的一个核心对象，可以通过crawler的settings属性访问配置文件

-  在from_crawler方法中，读取配置文件中的MONGO_DB_URI和MONGO_DB_NAME（不存在使用默认值），赋给cls的属性，即MongoDBPipeline类属性。
- 其他代码并没有任何改变，因为这里只是改变了设置MongoDBPipeline类属性的方式。

~~~python
# 在配置文件settings.py中对所要使用的数据库进行设置
MONGO_DB_URI = 'mongodb://192.168.1.105:27017/'
MONGO_DB_NAME = 'liushuo_scrapy_data'
~~~

------

## 第6章 使用LinkExtractor提取链接

在爬取一个网站时，想要爬取的数据通常分布在多个页面中，每个页面包含一部分数据以及到其他页面的链接，提取页面中数据的方法大家已经掌握，提取链接有使用**Selector**和使用**LinkExtractor**两种方法。

- **Selector**

  与提取数据方法相同，适用于**少量或规则简单的链接**提取

  ~~~python
  # 代码举例
  class BooksSpider(scrapy.Spider):
      ...
      def parse(self, response):
          ...
          # 提取链接
          # 下一页的url在ul.pager>li.next>a里面
          # 例如： <li class="next"><a href="catalogue/page-2.html">next</a></li>
          next_url = response.css('ul.pager li.next a::attr(href)').extract_first()
          if next_url:
              # 如果找到下一页的url，得到绝对路径，构造新的Request对象
              next_url = response.urljoin(next_url)
              yield scrapy.Request(next_url, callback=self.parse)
              ...
  ~~~

  

- **LinkExtractor**

  适用于提取**大量或规则复杂**的链接

  ~~~python
  # 代码举例
  from scrapy.linkextractors import LinkExtractor
  class BooksSpider(scrapy.Spider):
      ...
      def parse(self, response):
          ...
          # 提取链接
          # 下一页的url在ul.pager>li.next>a里面
          # 例如：<li class="next"><a href="catalogue/page-2.html">next</a></li>
          le = LinkExtractor(restrict_css='ul.pager li.next')
          links = le.extract_links(response)
          if links:
              next_url = links[0].url
              yield scrapy.Request(next_url, callback=self.parse)
  ~~~

  

### 6.1 使用LinkExtractor

~~~python
from scrapy.linkextractors import LinkExtractor
class BooksSpider(scrapy.Spider):
    ...
    def parse(self, response):
        ...
        # 提取链接
        # 下一页的url在ul.pager>li.next>a里面
        # 例如：<li class="next"><a href="catalogue/page-2.html">next</a></li>
        le = LinkExtractor(restrict_css='ul.pager li.next')
        links = le.extract_links(response)
        if links:
            next_url = links[0].url
            yield scrapy.Request(next_url, callback=self.parse)
~~~

对上述代码解释如下：

-  导入LinkExtractor，它位于scrapy.linkextractors模块。
- 创建一个LinkExtractor对象，使用一个或多个构造器参数描述提取规则，这里传递给restrict_css参数一个CSS选择器表达式。它描述出下一页链接所在的区域（在li.next下）。
-  调用LinkExtractor对象的extract_links方法传入一个Response对象，该方法依据创建对象时所描述的提取规则，在Response对象所包含的页面中提取链接，最终返回一个列表，其中的每一个元素都是一个Link对象，即提取到的一个链接。
- 由于页面中的下一页链接只有一个，因此用links[0]获取Link对象，Link对象的url属性便是链接页面的绝对url地址（无须再调用response.urljoin方法），用其构造Request对象并提交。

### 6.2 描述提取规则

~~~html
# 将下面两段html分别保存到文件example1.html和example2.html
<! -- example1.html -->
<html>
    <body>
        <div id="top">
            <p>下面是一些站内链接</p>
            <a class="internal" href="/intro/install.html">Installation guide</a>
            <a class="internal" href="/intro/tutorial.html">Tutorial</a>
            <a class="internal" href="../examples.html">Examples</a>
        </div>
        <div id="bottom">
            <p>下面是一些站外链接</p>
            <a href="http://stackoverflow.com/tags/scrapy/info">StackOverflow</a>
            <a href="https://github.com/scrapy/scrapy">Fork on Github</a>
        </div>
    </body>
</html>
<! -- example2.html -->
<html>
    <head>
        <script type='text/javascript' src='/js/app1.js'/>
            <script type='text/javascript' src='/js/app2.js'/>
    </head>
          <body>
            <a href="/home.html">主页</a>
            <a href="javascript:goToPage('/doc.html'); return false">文档</a>
            <a href="javascript:goToPage('/example.html'); return false">案例</a>
        </body>
</html>
~~~

~~~python
# 对以上2个HTML文本构造两个Response对象
>>> import os
>>> from scrapy.http import HtmlResponse
>>> from scrapy.linkextractors import LinkExtractor
>>> ex1 = 'D:\Python\pystation\scrapy\scrapy master\example\example1.html'
>>> ex2 = 'D:\Python\pystation\scrapy\scrapy master\example\example2.html'
>>> html1 = open(ex1,encoding='utf-8').read()
>>> html2 = open(ex2,encoding='utf-8').read()
>>> response1 = HtmlResponse(url='http://www.example1.com', body=html1, encoding='utf-8')
>>> response2 = HtmlResponse(url='http://www.example2.com', body=html2, encoding='utf-8')
>>> le = LinkExtractor()
>>> links = le.extract_links(response1)
>>> [link.url for link in links]
['http://www.example1.com/intro/install.html',
 'http://www.example1.com/intro/tutorial.html', 
 'http://www.example1.com/examples.html',
 'http://stackoverflow.com/tags/scrapy/info',
 'https://github.com/scrapy/scrapy']
~~~

LinkExtractor构造器的各个参数

- allow

  接收一个正则表达式或一个正则表达式列表，提取绝对url与正则表达式匹配的链接，如果该参数为空（默认），就提取全部链接。

  ~~~python
  # 提取页面example1.html中路径以/intro开始的链接
  >>> from scrapy.linkextractors import LinkExtractor
  >>> pattern = '/intro/.+\.html$'
  >>> le = LinkExtractor(allow=pattern)
  >>> links = le.extract_links(response1)
  >>> [link.url for link in links]
  ['http://example1.com/intro/install.html',
   'http://example1.com/intro/tutorial.html']
  ~~~

- deny

  接收一个正则表达式或一个正则表达式列表，与allow相反，排除绝对url与正则表达式匹配的链接。

  ~~~python
  # 提取页面example1.html中所有站外链接（即排除站内链接）
  >>> from scrapy.linkextractors import LinkExtractor
  >>> from urllib.parse import urlparse
  >>> pattern = patten = '^' + urlparse(response1.url).geturl()
  >>> pattern
  '^http://example1.com'
  >>> le = LinkExtractor(deny=pattern)
  >>> links = le.extract_links(response1)
  >>> [link.url for link in links]
  ['http://stackoverflow.com/tags/scrapy/info',
   'https://github.com/scrapy/scrapy']
  ~~~

- allow_domains

  接收一个域名或一个域名列表，提取到指定域的链接。

  ~~~python
  # 提取页面example1.html中所有到github.com和stackoverflow.com这两个域的链接
  >>> from scrapy.linkextractors import LinkExtractor
  >>> domains = ['github.com', 'stackoverflow.com']
  >>> le = LinkExtractor(allow_domains=domains)
  >>> links = le.extract_links(response1)
  >>> [link.url for link in links]
  ['http://stackoverflow.com/tags/scrapy/info',
   'https://github.com/scrapy/scrapy']
  ~~~

- deny_domains

  接收一个域名或一个域名列表，与allow_domains相反，排除到指定域的链接。

  ~~~python
  # 接收一个域名或一个域名列表，与allow_domains相反，排除到指定域的链接。
  >>> from scrapy.linkextractors import LinkExtractor
  >>> le = LinkExtractor(deny_domains='github.com')
  >>> links = le.extract_links(response1)
  >>> [link.url for link in links]
  ['http://example1.com/intro/install.html',
   'http://example1.com/intro/tutorial.html',
   'http://example1.com/../examples.html',
   'http://stackoverflow.com/tags/scrapy/info']
  ~~~

- restrict_xpaths

  接收一个XPath表达式或一个XPath表达式列表，提取XPath表达式选中区域下的链接。

  ~~~python
  # 提取页面example1.html中<div id="top">元素下的链接
  >>> from scrapy.linkextractors import LinkExtractor
  >>> le = LinkExtractor(restrict_xpaths='//div[@id="top"]')
  >>> links = le.extract_links(response1)
  >>> [link.url for link in links]
  ['http://example1.com/intro/install.html',
   'http://example1.com/intro/tutorial.html',
   'http://example1.com/../examples.html']
  ~~~

- restrict_css

  接收一个CSS选择器或一个CSS选择器列表，提取CSS选择器选中区域下的链接。

~~~python
# 提取页面example1.html中<div id="bottom">元素下的链接
>>> from scrapy.linkextractors import LinkExtractor
>>> le = LinkExtractor(restrict_css='div#bottom')
>>> links = le.extract_links(response1)
>>> [link.url for link in links]
['http://stackoverflow.com/tags/scrapy/info',
 'https://github.com/scrapy/scrapy']
~~~

- tags

  接收一个标签（字符串）或一个标签列表，提取指定标签内的链接，默认为['a', 'area']。

- attrs

  接收一个属性（字符串）或一个属性列表，提取指定属性内的链接，默认为['href']。

  ![image-20241018110303924](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101811030420241018110304.png)

  ~~~python
  # 提取页面example2.html中引用JavaScript文件的链接
  >>> from scrapy.linkextractors import LinkExtractor
  >>> le = LinkExtractor(tags='script', attrs='src')
  >>> links = le.extract_links(response2)
  >>> [link.url for link in links]
  ['http://example2.com/js/app1.js',
   'http://example2.com/js/app2.js']
  ~~~

- process_value

  接收一个形如func(value)的回调函数。如果传递了该参数，LinkExtractor将调用该回调函数对提取的每一个链接（如a的href）进行处理，回调函数正常情况下应返回一个字符串（处理结果），想要抛弃所处理的链接时，返回None。

  ~~~python
  # 在页面example2.html中，某些a的href属性是一段JavaScript代码，代码中包含了链接页面的实际url地址，此时应对链接进行处理，提取页面example2.html中所有实际链接
  >>> import re
  >>> def process(value):
      ...        m=re.search("javascript:goToPage\('(.*? )'", value)
      ...        # 如果匹配，就提取其中url并返回，不匹配则返回原值
      ...        if m:
          ...        value=m.group(1)
          ...        return value
      …
  >>> from scrapy.linkextractors import LinkExtractor
  >>> le = LinkExtractor(process_value=process)
  >>> links = le.extract_links(response2)
  >>> [link.url for link in links]
  ['http://example2.com/home.html',
   'http://example2.com/doc.html',
   'http://example2.com/example.html']
  ~~~

  ------

  

## 第7章 使用Exporter导出数据

在Scrapy中，负责导出数据的组件被称为Exporter（导出器）, Scrapy内部实现了多个Exporter，每个Exporter实现一种数据格式的导出，支持的数据格式如下（括号中为相应的Exporter）：

（1）JSON (JsonItemExporter)

（2）JSON lines (JsonLinesItemExporter)

（3）CSV (CsvItemExporter)

（4）XML (XmlItemExporter)

（5）Pickle (PickleItemExporter)

（6）Marshal (MarshalItemExporter)

### 7.1 指定如何导出数据

在导出数据时，需向Scrapy爬虫提供以下信息：

-  导出文件路径。
-  导出数据格式（即选用哪个Exporter）。

可以通过以下两种方式指定爬虫如何导出数据：

（1）通过命令行参数指定。

（2）通过配置文件指定。

#### 7.1.1 命令行参数

~~~python
$ scrapy crawl books -o books.csv
...
$head-10 books.csv   # 查看文件开头的10行
name, price
A Light in the Attic, £51.77
Tipping the Velvet, £53.74
Soumission, £50.10
Sharp Objects, £47.82
Sapiens: A Brief History of Humankind, £54.23
    The Requiem Red, £22.65
    The Dirty Little Secrets of Getting Your Dream Job, £33.34
    "The Coming Woman: A Novel Based on the Life of the Infamous Feminist, Victoria
    Woodhull", £17.93
    The Boys in the Boat: Nine Americans and Their Epic Quest for Gold at the 1936 Berlin
        Olympics, £22.60
~~~

-o books.csv：指定导出文件的路径

~~~python
$scrapy crawl books-t csv  -o books1.data
...
$ scrapy crawl books -t json -o books2.data
...
$scrapy crawl books-t xml  -o books3.data
...
~~~

-t：明确指出导出数据的格式。Scrapy爬虫通过文件后缀名推断出我们想以csv作为导出数据格式，所以不需要指定-t参数

指定导出文件路径时，还可以使用%(name)s和%(time)s两个特殊变量：

- %(name)s：会被替换为Spider的名字。
- %(time)s：会被替换为文件创建时间。

例如：假设一个项目中有爬取书籍信息、游戏信息、新闻信息的3个Spider，分别名为’books'、'games'、'news'。对于任意Spider的任意一次爬取，都可以使用'export_data/%(name)s/%(time)s.csv’作为导出路径，Scrapy爬虫会依据Spider的名字和爬取的时间点创建导出文件：

~~~python
$ scrapy crawl books -o 'export_data/%(name)s/%(time)s.csv'
...
$ scrapy crawl games -o 'export_data/%(name)s/%(time)s.csv'
...
$ scrapy crawl news -o 'export_data/%(name)s/%(time)s.csv'
...
$ scrapy crawl books -o 'export_data/%(name)s/%(time)s.csv'
...
$ tree export_data
export_data/
├── books
│  ├── 2017-03-06T02-31-57.csv
│  └── 2017-06-07T04-45-13.csv
├── games
│  └── 2017-04-05T01-43-01.csv
└── news
└── 2017-05-06T09-44-06.csv
~~~

#### 7.1.2 配置文件

使用配置文件可以配置导出文件路径、数据格式、文件编码、数据字段等功能

~~~python
# 导出文件路径
FEED_URI = 'export_data/%(name)s.data'
# 导出数据格式
FEED_FORMAT = 'csv'
# 导出文件编码
FEED_EXPORT_ENCODING = 'gbk'
# 导出数据包含的字段
FEED_EXPORT_FIELDS = ['name', 'author', 'price']
# 用户自定义Exporter字典，添加新的导出数据格式时使用
FEED_EXPORTERS = {'excel': 'my_project.my_exporters.ExcelItemExporter'}
~~~

### 7.2 添加导出数据格式

#### 7.2.2 实现Exporter

添加以Excel格式导出数据

~~~python
# 在项目中创建一个my_exporters.py（与settings.py同级目录），在其中实现ExcelItemExporter
from scrapy.exporters import BaseItemExporter
import xlwt
class ExcelItemExporter(BaseItemExporter):
    def __init__(self, file, **kwargs):
        self._configure(kwargs)
        self.file = file
        self.wbook = xlwt.Workbook()
        self.wsheet = self.wbook.add_sheet('scrapy')
        self.row = 0
    def finish_exporting(self):
        self.wbook.save(self.file)

    def export_item(self, item):
        fields = self._get_serialized_fields(item)
        for col, v in enumerate(x for _, x in fields):
            self.wsheet.write(self.row, col, v)
            self.row += 1
~~~

解释上述代码如下：

- 这里使用第三方库xlwt将数据写入Excel文件。
- 在构造器方法中创建Workbook对象和Worksheet对象，并初始化用来记录写入行坐标的self.row。
- 在export_item方法中调用基类的_get_serialized_fields方法，获得item所有字段的迭代器，然后调用self.wsheet.write方法将各字段写入Excel表格。
- finish_exporting方法在所有数据都被写入Excel表格后被调用，在该方法中调用self.wbook.save方法将Excel表格写入Excel文件。

完成ExcelItemExporter后，在配置文件settings.py中添加如下代码：

~~~python
FEED_EXPORTERS = {'excel': 'example.my_exporters.ExcelItemExporter'}
~~~

~~~python
# 以-t excel为参数，导出Excel格式的数据
$ scrapy crawl books -t excel -o books.xls
~~~

------

## 第8章 项目练习

![image-20241018150409809](C:\Users\Dawn\AppData\Roaming\Typora\typora-user-images\image-20241018150409809.png)

http://books.toscrape.com中书籍的详情页，列出了书籍的详细信息，包括

● 书名 √

● 价格 √

● 评价等级 √

● 书籍简介

● 产品编码 √

● 产品类型

● 税价

● 库存量 √

● 评价数量 √

我们对其中打勾的信息进行爬取

### 8.1 项目需求

爬取http://books.toscrape.com网站中的书籍信息。

（1）其中每一本书的信息包括：

➢ 书名

➢ 价格

➢ 评价等级

➢ 产品编码

➢ 库存量

➢ 评价数量

（2）将爬取的结果保存到csv文件中。

### 8.2 页面分析

~~~python
# 运行scrapy shell url命令
$ scrapy shell http://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html
    scrapy shell http://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html
        2017-03-03 09:17:01 [scrapy] INFO: Scrapy 1.3.3 started (bot: scrapybot)
                    2017-03-03 09:17:01 [scrapy] INFO: Overridden settings: {'LOGSTATS_INTERVAL': 0,
                                                                             'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter'}
                                    2017-03-03 09:17:01 [scrapy] INFO: Enabled extensions:
                                                    ['scrapy.extensions.corestats.CoreStats',
                                                     'scrapy.extensions.telnet.TelnetConsole']
                                                    2017-03-03 09:17:01 [scrapy] INFO: Enabled downloader middlewares:
                                                                    ['scrapy.downloadermiddlewares.httpauth.HttpAuthMiddleware',
                                                                     'scrapy.downloadermiddlewares.downloadtimeout.DownloadTimeoutMiddleware',
                                                                     'scrapy.downloadermiddlewares.defaultheaders.DefaultHeadersMiddleware',
                                                                     'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware',
                                                                     'scrapy.downloadermiddlewares.retry.RetryMiddleware',
                                                                     'scrapy.downloadermiddlewares.redirect.MetaRefreshMiddleware',
                                                                     'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware',
                                                                     'scrapy.downloadermiddlewares.redirect.RedirectMiddleware',
                                                                     'scrapy.downloadermiddlewares.cookies.CookiesMiddleware',
                                                                     'scrapy.downloadermiddlewares.chunked.ChunkedTransferMiddleware',
                                                                     'scrapy.downloadermiddlewares.stats.DownloaderStats']
                                                                    2017-03-03 09:17:01 [scrapy] INFO: Enabled spider middlewares:
                                                                                    ['scrapy.spidermiddlewares.httperror.HttpErrorMiddleware',
                                                                                     'scrapy.spidermiddlewares.offsite.OffsiteMiddleware',
                                                                                     'scrapy.spidermiddlewares.referer.RefererMiddleware',
                                                                                     'scrapy.spidermiddlewares.urllength.UrlLengthMiddleware',
                                                                                     'scrapy.spidermiddlewares.depth.DepthMiddleware']
                                                                                    2017-03-03 09:17:01 [scrapy] INFO: Enabled item pipelines:
                                                                                                    []
                                                                                                    2017-03-03 09:17:01 [scrapy] DEBUG: Telnet console listening on 127.0.0.1:6024
                                                                                                                    2017-03-03 09:17:01 [scrapy] INFO: Spider opened
                                                                                                                                2017-03-03 09:17:01 [scrapy] DEBUG: Crawled (200)  (referer: None)
                                                                                                                                            2017-03-03 09:17:02 [traitlets] DEBUG: Using default logger
        2017-03-03 09:17:02 [traitlets] DEBUG: Using default logger
                    [s] Available Scrapy objects:
                        [s]   scrapy    scrapy module (contains scrapy.Request, scrapy.Selector, etc)
                        [s]   crawler
                        [s]   item      {}
                        [s]   request
                        [s]   response   <200 http://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html>
                            [s]   settings
                            [s]   spider
                            [s] Useful shortcuts:
                                [s]   shelp()          Shell help (print this help)
                                [s]   fetch(req_or_url) Fetch request (or URL) and update local objects
                                [s]   view(response)   View response in a browser
                                >>>
~~~

运行这条命令后，scrapy shell会使用url参数构造一个Request对象，并提交给Scrapy引擎，页面下载完成后，程序进入一个python shell当中，在此环境中已经创建好了一些变量（对象和函数），以下几个最为常用：

- request

  最近一次下载对应的Request对象。

- response

  最近一次下载对应的Response对象

- fetch(req_or_url)

  该函数用于下载页面，可传入一个Request对象或url字符串，调用后会更新变量request和response

- view（response）

  用于在浏览器中显示response中的页面

  ~~~python
  # 输入view函数后会调用浏览器打开url
  >>> view(response)
  ~~~

![image-20241019171752707](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101917175920241019171759.png)

从图中可以看出，我们可在<div class="col-sm-6product_main">中提取书名、价格、评价等级，在scrapy shell中尝试提取这些信息

~~~python

~~~

