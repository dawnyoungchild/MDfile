



## 1、使用openpyxl处理xlsx文件

### 1.1 xlsx文件读取

~~~python
# 引入openpyxl模块，使用load_workbook方法打开xlsx文件
import openpyxl

wb = openpyxl.load_workbook('随便写.xlsx')

# 获取工作表名称
#获取当前工作表名称
currentSheet = wb.active
# 获取所有工作表名称,返回所有工作表名称组成的列表
allSheets = wb.get_sheet_names()
# 使用title属性获取当前工作表名称
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008545320230920085453.png" alt="image-20230905152755354" style="zoom: 50%;" />

### 1.2 设定当前工作表

~~~python
# 使用get_sheet_by_name()，变更当前工作表
ws = wb.get_sheet_by_name('人家是sheet3')
~~~

![image-20230905153107661](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008550120230920085501.png)

### 1.3 取得工作表和单元格内容

~~~python
# 获取单元格内容，ws[‘栏行'].value  # 栏是A, B, ……、行是1, 2, ……
import openpyxl

wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
print(ws['A1'].value)
~~~

![image-20230905153352022](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008550620230920085506.png)

~~~python
# 也可以使用cell().value方法取得单元格内容
wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
print(ws.cell(row=1, column=1).value)
'''
学校学校学校班级班级班级
'''
~~~



~~~python
# 获取单元格的相对位置，row行、column列和coordinate行列
wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
print(ws['C3'].row)
print(ws['C3'].column)
print(ws['C3'].coordinate)
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008551120230920085511.png" alt="image-20230905153700453" style="zoom: 67%;" />

~~~python
# 使用max_row和max_column获取工作表内容的行数和列数
wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
print('工作表行数：', ws.max_row)
print('工作表列数：', ws.max_column)
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008552020230920085520.png" alt="image-20230905154027807" style="zoom: 67%;" />

~~~python
# 获取整列或整行的内容
# ws.rows和ws.columns是生成器，可以将它们转换成列表形式以后获取到整行或整列的内容，通过value属性获取列表内元素的内容
wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
# 获取第一行单元格内容
for i in list(ws.rows)[0]:
    print(i.value)
'''
学校学校学校班级班级班级
班级
姓名
学号
年龄
'''
~~~

~~~python
# 列索引的数字字母转换
'''
在Excel中栏名称是A、B、…、Z、AA、AB、AC、……例如，1代表A、2代表B、26代表Z、27代表AA、28代表AB。如果工作表的栏数很多，很明显我们无法清楚了解到底索引是多少，例如，BC是多少。为了解决这方面的问题，下面将介绍2个转换方法：
'''
get_column_letter(数值)         # 将数值转成字母
column_index_from_string(字母)  # 将字母转成数值
# 上述方法存在于openpyxl.utils模块内，所以程序前面要加上下列指令。       
from openpyxl.utils import get_column_letter, column_index_from_string

# 举例
import openpyxl
from openpyxl.utils import get_column_letter, column_index_from_string

wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
print('3=', get_column_letter(3))
print('100=', get_column_letter(100))
print('E=', column_index_from_string('E'))
print('EEE=', column_index_from_string('EEE'))
'''
3= C
100= CV
E= 5
EEE= 3515
'''
~~~

~~~python
# 通过切片获取区间数据
wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
for row in ws['A2':'D3']:
    for cell in row:
        print(cell.value)
'''
一中
一·一
张三
01
二中
None
None
None
'''
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008552620230920085526.png" alt="image-20230905175351220" style="zoom:67%;" />

### 1.4 xlsx文件写入

#### 文件保存

~~~python
# 直接举例：建立空白工作簿（openpyxl.Workbook()），命名当前工作表为‘我被命名了’，然后保存到(save())‘测试新建.xlsx’
wb = openpyxl.Workbook()
ws = wb.active
ws.title = '我被命名了'
wb.save('测试新建.xlsx')
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008552920230920085529.png" alt="image-20230905175920892" style="zoom: 80%;" />

#### 复制Excel文件

~~~python
# 复制Excel文件：打开文件，然后用新名称保存文件实现文件复制。
wb = openpyxl.load_workbook('随便写.xlsx')
wb.save('我是另存的文件.xlsx')
~~~

#### 建立工作表

~~~python
# 建立工作表：使用creat_sheet()方法在工作簿里建立新工作表。
# 可以同时传入字符串，给新工作表自定义命名（第7行），也可以传入参数index,设置工作表在工作簿中的位置（第9行）。
wb = openpyxl.load_workbook('随便写.xlsx')
print(wb.get_sheet_names())
wb.create_sheet()
print(wb.get_sheet_names())
wb.create_sheet('onemore')
print(wb.get_sheet_names())
wb.create_sheet('onemoreagain', index=0)
print(wb.get_sheet_names())
wb.save('随便写2.xlsx')

'''
['Sheet1', '我是sheet2', '人家是sheet3']
['Sheet1', '我是sheet2', '人家是sheet3', 'Sheet']
['Sheet1', '我是sheet2', '人家是sheet3', 'Sheet', 'onemore']
['onemoreagain', 'Sheet1', '我是sheet2', '人家是sheet3', 'Sheet', 'onemore']
'''
~~~

#### 删除工作表

~~~python
# 使用remove_sheet()方法，必须调用get_sheet_by_name( )当作参数
import openpyxl

wb = openpyxl.load_workbook('随便写.xlsx')
print(wb.get_sheet_names())
wb.remove_sheet(wb.get_sheet_by_name('我是sheet2'))
print(wb.get_sheet_names())
'''
['Sheet1', '我是sheet2', '人家是sheet3']
['Sheet1', '人家是sheet3']
'''
~~~

#### 写入单元格

~~~python
# 写入单个单元格数据
# 输入数据的格式与在Excel窗口是相同的，字符串靠左对齐，数值数据靠右对齐。
import openpyxl

wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
ws['B3'] = '二·二'
wb.save('随便写.xlsx')
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008553520230920085535.png" alt="image-20230907155218007" style="zoom: 80%;" />

~~~python
# 写入列表数据
# 使用append()方法，可以将列表数据按照行写入，只能追加到单元格原有数据的下一行
wb = openpyxl.load_workbook('随便写.xlsx')
ws = wb.active
l1 = ['三中', '三·三', '李斯', '03', '19']
ws.append(l1)
wb.save('随便写2.xlsx')
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008553920230920085539.png" alt="image-20230907155753667" style="zoom:80%;" />

### 1.5 添加行列

~~~python
# 在指定index（索引从1开始）位置添加行列
# 在第4行插入1行空行
>>> sheet.insert_rows(4)
# 在第2行插入2行空行
>>> sheet.insert_rows(idx=2,amount=2)
# 添加一行数据到表追加到末尾行
>>> row_data = ["tom", 15, "tom@test.com"]
>>> sheet.append(row_data)
~~~





## 2.使用xlrd、xlwt处理xls文件

`xlrd` 用于读取文件，`xlwt` 用于写入文件,`xlutils` 是两个工具包的桥梁，也就是通过xlrd 读取`.xls`文件，然后通过xlutils 将文件内容交给xlwt处理并且保存。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023092008554420230920085544.png" alt="image-20230913172502229" style="zoom: 67%;" />

下面举例都是对上图表格进行操作

### 2.1 xlrd读取数据

~~~python
#导入
import xlrd
# 打开文件 必须是存在的文件路径
wb = xlrd.open_workbook('路径')
# 获取文件中所有的sheet对象
objects = wb.sheets()
#获取文件中所有的sheet名称
names = wb.sheet_names()
# 按照索引获得sheet对象
ws = wb.sheet_by_index(索引值)
#按照名称获得sheet对象
ws = wb.sheet_by_name(文件名)
#获得当前sheet对象的名称
name = ws.name
'''
202302
'''
#获得当前excel文件的sheet个数
n = wb.nsheets
#获得当前sheet已使用的行和列
nrows = ws.nrows
ncols = ws.ncols
'''
118
12
'''
# 获得当前sheet某一行或者某一列的所有元素 元素格式是：数据类型：数据值
# 数据类型：0.空，1.字符串，2.数字，3.日期，4.布尔，5.error
# 返回的是列表格式
lst = ws.row(索引值)
lst = ws.col(索引值)
'''
获取第30行数据：lst = ws.row(29)
[text:'中国广电山东网络有限公司潍坊市分公司', 
 empty:'', 
 text:'3700576214779', 
 text:'潍坊市青州市邵庄镇邵庄村439号', 
 empty:'', 
 empty:'', 
 text:'一般工商业', 
 empty:'', 
 number:26.0, 
 empty:'', 
 number:18.49, 
 text:'邵庄']
'''
# 是ws.row(索引值)和s.col(索引值)得到允许切片版本
lst = ws.row_slice(索引值, start_colx=0, end_colx=None)
lst = ws.col_slice(索引值, start_rowx=0, end_rowx=None)
'''
获取C列从30行到最后一行的数据：lst = ws.col_slice(2, start_rowx=29, end_rowx=None)
[text:'3700576214779', text:'3700061508486', text:'3700701058339', text:'3700576013943', text:'3701054806831', ......]
'''

# 获得当前sheet某一行或者某一列的所有元素的值，可以指定起止索引
lst = ws.row_values(索引值)
lst = ws.col_values(索引值)
'''
获取第30行数据：lst = ws.row_values(29)
['中国广电山东网络有限公司潍坊市分公司', '', '3700576214779', '潍坊市青州市邵庄镇邵庄村439号', '', '', '一般工商业', '', 26.0, '', 18.49, '邵庄']

获取C列数据，从30行开始：lst = ws.col_values(2, start_rowx=29)
['3700576214779', '3700061508486', '3700701058339', '3700576013943', '3701054806831', '3700849609961', '3700782312083', '3700847356490', ......]
'''
# 获得当前sheet某一行或者某一列的所有元素的数据类型
lst = ws.row_types(索引值，start_colx=0, end_colx=None)
lst = ws.col_types(索引值, start_rowx=0, end_rowx=None)
# 返回当前sheet某一行已使用的长度
i = ws.row_len(索引值)
# 获取当前sheet中某个单元格的元素 元素格式是：数据类型：数据值
# 数据类型：0.空，1.字符串，2.数字，3.日期，4.布尔，5.error
n = ws.cell(行索引， 列索引)
'''
获取A列30行单元格数据：n = ws.cell(29, 0)
text:'中国广电山东网络有限公司潍坊市分公司'
'''
# 获取当前sheet中某个单元格的元素的值
n = ws.cell_value(行索引， 列索引)
n = ws.cell(行索引，列索引).value
n = ws.row(行索引)[列索引].value
'''
获取A列30行单元格数据：n = ws.cell_value(29, 0)
中国广电山东网络有限公司潍坊市分公司
'''
# 获取当前sheet中某个单元格的元素的数据类型
n = ws.cell_type(行索引， 列索引)
n = ws.cell(行索引，列索引).ctype
n = ws.row(行索引)[列索引].ctype
~~~

### 2.2 xlwt写入数据

~~~python
# 导入
import xlwt
# 新建.xls的文件
nwb = xlwt.Workbook('utf-8')
# 添加工作表
nws = nwb.add_sheet('名称', cell_overwrite_ok=True)
# 在对应单元格上写入内容
nws.write(行索引，列索引，写入内容)
# 保存文件 注意.xls的后缀不能少
nwb.save('文件名.xls')
~~~

### 2.3 通过xlutils模块同时读取和写入数据

~~~python
# 导入
from xlutils.copy import copy
# 用xlrd导入待处理的文件
wb = xlrd.open_workbook('文件名')
#使用copy函数处理 此时的 nwb 不仅具有 xlwt 的功能, 还具有 xlrd 的功能
#也就是既可读又可写.
nwb = copy(wb)
~~~

