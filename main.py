import re, os, psycopg2, datetime, importlib
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
#from dotenv import load_dotenv
import segmentação.inicializador as init

importlib.reload(init)

service, options = init.start_driver()

year = 2007

# Inicia-se a instância do Chrome WebDriver com as definidas 'options' e 'service', basicamente o driver É o google chrome.
for i in range(20):
    year = year + 1
    print(f'Coletando dados do ano {year}...')
    driver = init.go_to_site(service, options, f'https://www.boxofficemojo.com/year/world/{year}/')

    table = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, "//table[@class='a-bordered a-horizontal-stripes a-size-base a-span12 mojo-body-table mojo-table-annotated scrolling-data-table']/tbody/tr/td/a[@href]")))
    links = [element.get_attribute("href") for element in table]

    lista_geral = []

    # Clica no primeiro link da lista e abre a página correspondente
    for link in links:

        driver.get(link)

        # pega o link do sumário (que abre a outra página com as informações que queremos)
        try:
            title_summary_link = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, "//div[@id='title-summary-refiner']/a")))
        except:
            continue
        # espera até que o link esteja presente e clica nele
        try:
            title_summary_link.click()
        except:
            print(f'Erro ao clicar no link: {link}')
            continue

        info_dict = {}
        # coleta do título, resumo e bilheteria (nacional, internacional e mundial)
        info_dict['movie_title'] = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, "//h1[@class='a-size-extra-large']"))).text
        info_dict['movie_summary'] = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, "//span[@class='a-size-medium']"))).text
        gross = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
            (By.XPATH, "//div[@class='a-section a-spacing-none mojo-performance-summary-table']/div/span[@class='a-size-medium a-text-bold']")))
        if gross[0].text == '–':
            info_dict['international_gross'] = int(gross[1].text.replace('$', '').replace(',', ''))
            info_dict['total_gross'] = int(gross[2].text.replace('$', '').replace(',', ''))
        elif gross[1].text == '–':     
            info_dict['domestic_gross'] = int(gross[0].text.replace('$', '').replace(',', ''))
            info_dict['total_gross'] = int(gross[2].text.replace('$', '').replace(',', ''))
        else:
            info_dict['domestic_gross'] = int(gross[0].text.replace('$', '').replace(',', ''))
            info_dict['international_gross'] = int(gross[1].text.replace('$', '').replace(',', ''))
            info_dict['total_gross'] = int(gross[2].text.replace('$', '').replace(',', ''))

        # coleta do ano de lançamento, classificação etária, gênero, data de lançamento e duração
        table_info = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, "//div[@class='a-section a-spacing-none mojo-summary-values mojo-hidden-from-mobile']/div"))) 

        # essa outra tabela vem separada por \n, então precisamos separar as linhas
        # e depois separar por chave e valor (achei mais simples assim)
        for i in table_info:
            lines = i.text.split('\n')
            if len(lines) >= 2:
                key = lines[0].strip()
                value = lines[1].strip()
                info_dict[key] = value

        # ajuste de data para datetime (imagino que vá facilitar no banco de dados)
        # Remove the parenthesis and extra info
        date_str = info_dict['Earliest Release Date'].split('(')[0].strip()

        # Map Portuguese month names to English
        month_map = {
            'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April',
            'Maio': 'May', 'Junho': 'June', 'Julho': 'July', 'Agosto': 'August',
            'Setembro': 'September', 'Outubro': 'October', 'Novembro': 'November', 'Dezembro': 'December'
        }

        for pt, en in month_map.items():
            if pt in date_str:
                date_str = date_str.replace(pt, en)
                break

        info_dict['Earliest Release Date'] = datetime.datetime.strptime(date_str, '%B %d, %Y').date().isoformat()

        # domestic open (ajuste para int e remover o $ e ,)
        try:
            info_dict['Domestic Opening'] = int(info_dict['Domestic Opening'].replace('$', '').replace(',', ''))
        except KeyError:
            pass

        if 'Budget' in info_dict:
            info_dict['Budget'] = int(info_dict['Budget'].replace('$', '').replace(',', ''))



        # running time (ajuste para int e remover o hr e min e transformar em minutos)
        try:
            split_running_time = info_dict['Running Time'].split(' ')
        except KeyError:
            info_dict['Running Time'] = None

        try:
            liquid_running_time = {split_running_time[1]: split_running_time[0], split_running_time[3]: split_running_time[2]}
            minutes = True
        except:
            try:
                liquid_running_time = {split_running_time[1]: split_running_time[0]}
                minutes = False
            except:
                minutes = False

        if minutes:
            info_dict['Running Time'] = int(liquid_running_time['hr']) * 60 + int(liquid_running_time['min'])
        else:
            try:
                info_dict['Running Time'] = int(liquid_running_time['hr']) * 60
            except:
                try:
                    info_dict['Running Time'] = int(liquid_running_time['min'])
                except KeyError:
                    info_dict['Running Time'] = None

        info_dict.pop('IMDbPro', None)

        lista_geral.append(info_dict)

    df = pd.DataFrame(lista_geral)
    df.pop('Budget')
    #df.pop('MPAA')

    df.rename(columns={'Domestic Distributor': 'domestic_distributor'}, inplace=True)
    df.rename(columns={'Domestic Opening': 'domestic_opening'}, inplace=True)
    df.rename(columns={'Earliest Release Date': 'release_date'}, inplace=True)
    df.rename(columns={'Running Time': 'running_time'}, inplace=True)
    df.rename(columns={'Genres': 'genres'}, inplace=True)
    df.rename(columns={'MPAA': 'indicative_rating'}, inplace=True)

    df = df[['movie_title', 'movie_summary', 'release_date', 'genres', 'indicative_rating', 'running_time', 'domestic_gross', 'international_gross', 'total_gross', 'domestic_opening', 'domestic_distributor']]

    df.to_csv(f'worldwide_box_office_{year}.csv', index=False)

    driver.quit()