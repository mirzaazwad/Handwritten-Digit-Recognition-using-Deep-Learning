# Code analysis
## Handwritten-Digit-Recognition-using-Deep-Learning 
#### Branch main
#### Version 1.0 

**By: Administrator**

*Date: 2025-06-19*

*Analyzed the: 2025-06-19*

## Introduction
This document contains results of the code analysis of Handwritten-Digit-Recognition-using-Deep-Learning



## Configuration

- Quality Profiles
    - Names: Sonar way [Python]; 
    - Files: 8dd42e22-6dfb-4642-b0cc-2fca2ac2e0d0.json; 


 - Quality Gate
    - Name: Sonar way
    - File: Sonar way.xml

## Synthesis

### Analysis Status

Reliability | Security | Security Review | Maintainability |
:---:|:---:|:---:|:---:
A | A | A | A |

### Quality gate status

| Quality Gate Status | OK |
|-|-|



### Metrics

Coverage | Duplications | Comment density | Median number of lines of code per file | Adherence to coding standard |
:---:|:---:|:---:|:---:|:---:
0.0 % | 59.6 % | 9.7 % | 57.0 | 99.8 %

### Tests

Total | Success Rate | Skipped | Errors | Failures |
:---:|:---:|:---:|:---:|:---:
0 | 0 % | 0 | 0 | 0

### Detailed technical debt

Reliability|Security|Maintainability|Total
---|---|---|---
-|-|0d 1h 12min|0d 1h 12min


### Metrics Range

\ | Cyclomatic Complexity | Cognitive Complexity | Lines of code per file | Coverage | Comment density (%) | Duplication (%)
:---|:---:|:---:|:---:|:---:|:---:|:---:
Min | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0
Max | 13.0 | 11.0 | 134.0 | 0.0 | 32.9 | 98.7

### Volume

Language|Number
---|---
Python|541
Total|541


## Issues

### Issues count by severity and types

Type / Severity|INFO|MINOR|MAJOR|CRITICAL|BLOCKER
---|---|---|---|---|---
BUG|0|0|0|0|0
VULNERABILITY|0|0|0|0|0
CODE_SMELL|0|1|14|0|0


### Issues List

Name|Description|Type|Severity|Number
---|---|---|---|---
Sections of code should not be commented out||CODE_SMELL|MAJOR|7
Results that depend on random number generation should be reproducible||CODE_SMELL|MAJOR|2
numpy.random.Generator should be preferred to numpy.random.RandomState||CODE_SMELL|MAJOR|4
Important hyperparameters should be specified for machine learning libraries' estimators and optimizers||CODE_SMELL|MAJOR|1
Local variable and function parameter names should comply with a naming convention||CODE_SMELL|MINOR|1


## Security Hotspots

### Security hotspots count by category and priority

Category / Priority|LOW|MEDIUM|HIGH
---|---|---|---
LDAP Injection|0|0|0
Object Injection|0|0|0
Server-Side Request Forgery (SSRF)|0|0|0
XML External Entity (XXE)|0|0|0
Insecure Configuration|0|0|0
XPath Injection|0|0|0
Authentication|0|0|0
Weak Cryptography|0|0|0
Denial of Service (DoS)|0|0|0
Log Injection|0|0|0
Cross-Site Request Forgery (CSRF)|0|0|0
Open Redirect|0|0|0
Permission|0|0|0
SQL Injection|0|0|0
Encryption of Sensitive Data|0|0|0
Traceability|0|0|0
Buffer Overflow|0|0|0
File Manipulation|0|0|0
Code Injection (RCE)|0|0|0
Cross-Site Scripting (XSS)|0|0|0
Command Injection|0|0|0
Path Traversal Injection|0|0|0
HTTP Response Splitting|0|0|0
Others|0|0|0


### Security hotspots


