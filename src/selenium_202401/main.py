import load_html
import load_url


def main():
    load_url.fetch_url()
    year = 2019
    month = 1
    load_html.fetch_html(year, month)


if __name__ == "__main__":
    main()
