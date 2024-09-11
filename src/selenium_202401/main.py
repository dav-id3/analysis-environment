import load_html
import load_url
import load_data


def main():
    start_year = 2021
    start_month = 1
    # load_url.fetch_url(start_year, start_month)

    # load_html.request_html(start_year, start_month)

    # load_data.request_data(start_year, start_month)

    load_data.request_data(start_year, start_month)


if __name__ == "__main__":
    main()
