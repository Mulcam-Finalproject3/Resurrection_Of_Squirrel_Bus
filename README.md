# 프로젝트 제목
수요 대응 버스 분석 및 노선 신설 방안 제시 

## 프로젝트 개요
> 운행중인 다람쥐 버스의 특성을 분석하여, 다람쥐 버스가 필요한 새로운 지역을 선정하고 경로를 설정하는 프로젝트 입니다.

## 프로젝트 내
1. 데이터 추출
2. EDA
3. 지역 분석
4. 지역별 다람쥐 버스 노선 분석
5. 

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

# OpenDataWrangling
* 공공데이터 분석
* 다음의 버전을 사용했습니다.
* !pip install plotnine==0.5.0
* !pip install --upgrade pandas==0.23.4
* !pip install folium==0.5.0
* :tv: [전체영상 보기](https://goo.gl/TJeiTi)

## 전체 강좌 보기
* 인프런 : [https://www.inflearn.com/course/공공데이터로-파이썬-데이터-분석-시작하기]( https://www.inflearn.com/course/%EA%B3%B5%EA%B3%B5%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A1%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%84%EC%84%9D-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0#)

* 아래 튜토리얼은 다음의 저장소에 코드와 설명을 보완해서 새로 올렸습니다.
	* [corazzon/open-data-analysis-basic: 공공데이터로 데이터 분석 시작하기 기초](https://github.com/corazzon/open-data-analysis-basic)

## [스타벅스, 이디야 분석](store_location_by_folium.ipynb)

* 이디야는 스타벅스 근처에 입점한다는 설이 있었습니다. 과연 이디야와 스타벅스의 매장입지는 얼마나 차이가 날까요? 관련 기사를 읽고 구별로 이디야와 스타벅스의 매장을 기사와 유사하게 분석하고 시각화 해보면서 Python, Pandas, ggplot(plotnine), Numpy, Folium에 익숙해져 봅니다.
* 인용기사 : [[비즈&빅데이터]스타벅스 '쏠림' vs 이디야 '분산'](http://news.bizwatch.co.kr/article/consumer/2018/01/19/0015)
* 공공데이터 포털에서 제공하고 있는 소상공인시장진흥공단 상가업소정보 데이터를 분석합니다. 대분류와 중분류 데이터를 Pandas, ggplot, folium으로 분석합니다.
* 데이터 출처 : [https://www.data.go.kr/dataset/15012005/fileData.do](https://www.data.go.kr/dataset/15012005/fileData.do)





![](../header.png)
---
## 목차 
- 개요
- 



## 개요


OS X & 리눅스:

```sh
npm install my-crazy-module --save
```

윈도우:

```sh
edit autoexec.bat
```

## 사용 예제

스크린 샷과 코드 예제를 통해 사용 방법을 자세히 설명합니다.

_더 많은 예제와 사용법은 [Wiki][wiki]를 참고하세요._

## 개발 환경 설정

모든 개발 의존성 설치 방법과 자동 테스트 슈트 실행 방법을 운영체제 별로 작성합니다.

```sh
make install
npm test
```

## 업데이트 내역

* 0.2.1
    * 수정: 문서 업데이트 (모듈 코드 동일)
* 0.2.0
    * 수정: `setDefaultXYZ()` 메서드 제거
    * 추가: `init()` 메서드 추가
* 0.1.1
    * 버그 수정: `baz()` 메서드 호출 시 부팅되지 않는 현상 (@컨트리뷰터 감사합니다!)
* 0.1.0
    * 첫 출시
    * 수정: `foo()` 메서드 네이밍을 `bar()`로 수정
* 0.0.1
    * 작업 진행 중

## 정보

이름 – [@트위터 주소](https://twitter.com/dbader_org) – 이메일주소@example.com

XYZ 라이센스를 준수하며 ``LICENSE``에서 자세한 정보를 확인할 수 있습니다.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## 기여 방법

1. (<https://github.com/yourname/yourproject/fork>)을 포크합니다.
2. (`git checkout -b feature/fooBar`) 명령어로 새 브랜치를 만드세요.
3. (`git commit -am 'Add some fooBar'`) 명령어로 커밋하세요.
4. (`git push origin feature/fooBar`) 명령어로 브랜치에 푸시하세요. 
5. 풀리퀘스트를 보내주세요.

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
