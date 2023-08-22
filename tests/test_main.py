from main import *

def test_has_ipv4():
    """
    Test if a domain has an IPv4 address in it separated by a dot or a dash.
    """
    assert PRCluster.has_ipv4("192.168.0.1.example.com")
    assert PRCluster.has_ipv4("192-168-0-1.example.com")
    assert not PRCluster.has_ipv4("192-168.example.com")
                              
def test_has_32hexa():
    """
    Test if a domain has a 32-hexadecimal string in it.
    """

    assert PRCluster.has_32hexa("0123456789abcdef0123456789abcdef.example.com")
    assert not PRCluster.has_32hexa("x123456789abcdef0123456789abcdef.example.com")
    assert not PRCluster.has_32hexa("0123456789abcdef.example.com")

def test_has_uuid():
    """
    Test if a domain has a UUID in it.
    """    
    assert PRCluster.has_uuid("01234567-89ab-cdef-0123-456789abcdef.example.com")
    assert not PRCluster.has_uuid("01234567-89ab-cdef-0123.example.com")

def test_exclude_domains():
    """
    Test if a domain should be excluded.
    """
    assert PRCluster.exclude_domains(["example.com"]) == ["example.com"]
    # invalid domain name
    assert PRCluster.exclude_domains(["example.exampleexample"]) == []
    assert PRCluster.exclude_domains(["192.168.1.1"]) == []
    # too long domain name
    assert PRCluster.exclude_domains(["exampleexampleexampleexampleexampleexampleexampleexampleexampleexampleexampleexample.example"]) == []
    # more than half of 2ld is number
    assert PRCluster.exclude_domains(["01234567890123456789.example.com"]) == []

def test_get_common_part():
    """
    Test if a common part of two strings can be obtained.
    """
    assert PRDetector.get_common_part("www1.example.com", "www2.example.com") == ".example.com"
    assert PRDetector.get_common_part("example.com", "111example.com") == "example.com"
    assert PRDetector.get_common_part("example.com", "example111.com") == ".com"
    assert PRDetector.get_common_part("example.com", "example111.net") == ""

def test__filter_out_domain():
    """
    Test if a domain matches filters.
    """
    assert PRDetector._filter_out_domain("example.com", [{"type":"length","val":11}]) is False
    assert PRDetector._filter_out_domain("example.com", [{"type":"length","val":10}]) is True
    assert PRDetector._filter_out_domain("example.com", [{"type":"tld","val":"com"}]) is False
    assert PRDetector._filter_out_domain("example.com", [{"type":"tld","val":"net"}]) is True
    assert PRDetector._filter_out_domain("www.example.com", [{"type":"fld","val":"example.com"}]) is False
    assert PRDetector._filter_out_domain("www.example.net", [{"type":"fld","val":"example.com"}])