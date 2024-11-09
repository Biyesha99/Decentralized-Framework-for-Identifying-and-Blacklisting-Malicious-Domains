pragma solidity ^0.8.0;

contract MaliciousDomainRegistry {
    struct Domain {
        string url;
        string ip;
        bool isMalicious;
    }

    mapping(string => Domain) public domains;

    function addDomain(string memory _url, string memory _ip, bool _isMalicious) public {
        domains[_url] = Domain(_url, _ip, _isMalicious);
    }

    function getDomain(string memory _url) public view returns (string memory, string memory, bool) {
        Domain memory domain = domains[_url];
        return (domain.url, domain.ip, domain.isMalicious);
    }
}
