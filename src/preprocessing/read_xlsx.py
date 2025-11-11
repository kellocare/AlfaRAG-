import pandas as pd

def read_column_to_list(excel_file, column=0, sheet_name=0, header=0, skip_rows=0):
    try:
        df = pd.read_excel(excel_file, 
                          sheet_name=sheet_name, 
                          header=header,
                          skiprows=skip_rows)
        
        if isinstance(column, int):
            selected_column = df.iloc[:, column].tolist()
        elif isinstance(column, str):
            selected_column = df[column].tolist()
        else:
            raise ValueError("Параметр 'column' должен быть int (индекс) или str (имя столбца)")
        
        return selected_column
    
    except FileNotFoundError:
        print(f"Ошибка: Файл '{excel_file}' не найден")
        return []
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return []

if __name__ == "__main__":
    # Примеры использования:    
    data_column_list = read_column_to_list("excel/websites_updated.xlsx", column=1)
    print(f"Получено {len(data_column_list)} элементов:")
    for i, item in enumerate(data_column_list[:2], 1):
        print(f"{i}: {item}")
    