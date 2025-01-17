# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import hashlib
import os
import math
import platform
import random
import socket
import time

import bytedtrace
from email.utils import parsedate_tz
import crcmod
import sys

from bytedtos.models import TaggingRule, to_put_tagging

try:
    from urllib.parse import quote
except:
    from urllib import quote

import requests
import six
from servicediscovery import lookup_name
from six.moves.urllib.parse import urlencode
from requests.structures import CaseInsensitiveDict

from bytedtos.__version__ import __version__
from bytedtos.credentials import Credentials, BucketAccessKeyCredentials
from bytedtos.errors import TosException
from bytedtos.sign import SignVersion, PublicSigner, PlainSigner
from bytedtos.sign_v1 import SignerV1
from bytedtos.sign_v4 import SignerV4
from bytedtos.consts import *
from bytedtos.cn_consts.cn_consts import CN_AVAILABLE_ENDPOINTS
from bytedtos.i18n_consts.i18n_consts import I18N_AVAILABLE_ENDPOINTS


DEFAULT_TIMEOUT = 10
DEFAULT_CONNECT_TIMEOUT = 3
DEFAULT_CLUSTER = "default"

TOS_API_SERVICE_NAME = "toutiao.tos.tosapi"
TOS_ACCESS_HEADER = "x-tos-access"
TOS_MD5_HEADER = "x-tos-md5"
TOS_ETAG_HEADER = "x-tos-etag"
HTTP_USER_AGENT_HEADER = "User-Agent"


def is_ipv6(ip):
    try:
        socket.inet_pton(socket.AF_INET6, ip)
    except socket.error:
        return False
    return True


def parse_date(ims):
    if not ims:
        return 0
    """ Parse rfc1123, rfc850 and asctime timestamps and return UTC epoch. """
    try:
        ts = parsedate_tz(ims)
        return int(time.mktime(ts[:8] + (0,)) - (ts[9] or 0) - time.timezone)
    except (TypeError, ValueError, IndexError, OverflowError):
        return 0


def md5data(data):
    buf_size = 1024 * 1024  # 1MB
    md5 = hashlib.md5()
    if hasattr(data, "read") and hasattr(data, "seek"):
        buf = data.read(buf_size)
        while len(buf) > 0:
            md5.update(buf)
            buf = data.read(buf_size)
        data.seek(0, 0)
    else:
        md5.update(data)
    if not isinstance(data, six.binary_type):
        raise ValueError("data type is %s, %s expected", type(data), six.binary_type)
    return md5.hexdigest()


def crc64data(data):
    crc = crcmod.Crc(0x142F0E1EBA9EA3693, initCrc=0, xorOut=0XFFFFFFFFFFFFFFFF, rev=True)
    buf_size = 1024 * 1024  # 1MB
    if hasattr(data, "read") and hasattr(data, "seek"):
        buf = data.read(buf_size)
        while len(buf) > 0:
            crc.update(buf)
            buf = data.read(buf_size)
        data.seek(0, 0)
    else:
        crc.update(data)
    if not isinstance(data, six.binary_type):
        raise ValueError("data type is %s, %s expected", type(data), six.binary_type)
    return str(crc.crcValue)


def assert_validate_key(k):
    if not k or k.startswith("/") or k.endswith("/") or "//" in k or "/../" in k or "/./" in k:
        raise ValueError("invalid key: %s" % k)


def make_signer(credentials, sign_version, region):
    """
    :type credentials: Credentials
    :type sign_version: SignVersion
    :type region: str
    :rtype: Signer
    """
    if sign_version == SignVersion.Public:
        return PublicSigner()
    if sign_version == SignVersion.Plain:
        return PlainSigner(credentials.get_access_key(), credentials.get_secret_key())
    if sign_version == SignVersion.V4:
        if not region:
            raise ValueError('region must be set when using SignatureV4')
        return SignerV4(credentials.get_access_key(), credentials.get_secret_key(), region)
    return SignerV1(credentials.get_access_key(), credentials.get_secret_key(), region=region)



def path_style(bucket, key):
    """
    :type bucket: str
    :type key: str
    :rtype: str
    """
    path = '/' + bucket
    if key:
        path = path + '/' + key
    return path


def _is_func_can_retry(fun_name):
    if fun_name in WHITE_LIST_FUNCTION:
        return True
    return False


class Response(object):
    def __init__(self, res, part_number=None):
        self._resp = res
        self._part_number = part_number

    @property
    def json(self):
        return self._resp.json()

    @property
    def headers(self):
        return self._resp.headers

    @property
    def data(self):
        return self._resp.content

    @property
    def size(self):
        return int(self.headers.get("content-length"))

    @property
    def last_modify_time(self):
        return parse_date(self.headers.get("last-modified"))

    @property
    def upload_id(self):
        return self.json["payload"]["uploadID"]

    @property
    def part_number(self):
        if not self._part_number:
            raise ValueError("part_number not exist")
        if self.headers.get(TOS_ETAG_HEADER):
            return self._part_number + ":" + self.headers.get(TOS_ETAG_HEADER)
        return self._part_number

    @property
    def status_code(self):
        return self._resp.status_code

    @property
    def raw(self):
        return self._resp.raw


class Client(object):

    def __init__(self, bucket, access_key, **kwargs):
        self.bucket = bucket
        if isinstance(access_key, Credentials):
            self.credentials = access_key
        else:
            self.credentials = BucketAccessKeyCredentials(bucket, access_key)
        self.user_agent = self._get_user_agent()
        self.timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
        self.connect_timeout = kwargs.get("connect_timeout", DEFAULT_CONNECT_TIMEOUT)
        self.cluster = kwargs.get("cluster", DEFAULT_CLUSTER)
        self.endpoint = kwargs.get("endpoint")
        self.addrs = kwargs.get("addrs")
        self.service_name = kwargs.get("service", TOS_API_SERVICE_NAME)
        self.idc = kwargs.get("idc", "")
        self.stream = kwargs.get("stream", False)
        self.region = kwargs.get('region', '')
        self.remote_psm = kwargs.get('remote_psm', os.environ.get('TCE_PSM') or os.environ.get('LOAD_SERVICE_PSM') or os.environ.get('PSM'))
        self.sign_version = kwargs.get('sign_version', SignVersion.V1)
        self.enable_https = kwargs.get('enable_https', False)
        self.enable_crc64 = kwargs.get('enable_crc64', False)
        self.force_endpoint = kwargs.get('force_endpoint', False)
        self.session = requests.Session()
        connection_pool_size = kwargs.get('connection_pool_size', 10)
        self.session.mount('http://', requests.adapters.HTTPAdapter(pool_connections=connection_pool_size,
                                                                    pool_maxsize=connection_pool_size,
                                                                    max_retries=3))
        self.session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=connection_pool_size,
                                                                     pool_maxsize=connection_pool_size,
                                                                     max_retries=3))
        self.disable_bns = kwargs.get("disable_bns", False)
        self.max_retry_count = kwargs.get("max_retry_count", 3)
        tos_force_psm = os.environ.get("TOS_FORCE_PSM")
        if tos_force_psm and tos_force_psm.lower() == "true":
            self.disable_bns = True
        if self.endpoint and not self._is_endpoint_valid():
            raise ValueError("Invalid endpoints")
        if not self.endpoint and self.enable_https:
            raise ValueError("HTTPS is only available with endpoints")
        if not kwargs.get('using_signature', True):  # same as other SDKs
            self.sign_version = SignVersion.Plain
        self.signer = make_signer(self.credentials, self.sign_version, self.region)
        
        self.rd = random.Random()

    def _read(self, data):
        if hasattr(data, "read") and hasattr(data, "seek"):
            d = data.read()
            data.seek(0, 0)
        else:
            d = data
        if isinstance(d, six.binary_type):
            return d
        if isinstance(d, six.text_type):
            return d.encode()

        raise ValueError("don't support type %s", type(data))

    def put_object(self, key, data, headers=None):
        assert_validate_key(key)
        md5hash = None
        crc64W = None
        if self.enable_crc64:
            crc64W = crc64data(self._read(data))
        else:
            md5hash = md5data(self._read(data))

        res = self._req("PUT", key, data, headers=headers, client_method=PutObjectMethod)
        if res.status_code != 200:
            raise TosException(res)

        if crc64W is not None and self.enable_crc64 \
                and res.headers.get(TOS_HASH_CRC64_ECMA) is not None \
                and res.headers.get(TOS_HASH_CRC64_ECMA) != ""\
                and res.headers.get(TOS_HASH_CRC64_ECMA) != crc64W:
            raise TosException(res, "expexct: %s; actual: %s" % (crc64W, res.headers.get(TOS_HASH_CRC64_ECMA)))
        if md5hash is not None \
                and res.headers.get(TOS_MD5_HEADER) is not None \
                and res.headers.get(TOS_MD5_HEADER) != "" \
                and res.headers.get(TOS_MD5_HEADER) != md5hash:
            raise TosException(res, "expect:%s; actual:%s" % (md5hash, res.headers.get(TOS_MD5_HEADER)))
        return Response(res)

    def get_object(self, key, version_id=None):
        assert_validate_key(key)
        query = None
        if version_id:
            query = {'versionId': version_id}
        res = self._req("GET", key, query=query, client_method=GetObjectMethod)
        if res.status_code == 404:
            raise TosException(res, "Not Found")
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def get_object_range(self, key, start, end):
        assert_validate_key(key)
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("start and end should be int")
        if start > end:
            raise ValueError("start should not be larger than end, start is %d, end is %d" % (start, end))

        res = self._req("GET", key, headers={"Range": "bytes=" + str(start) + "-" + str(end)},
                        client_method=GetObjectFromRangeMethod)
        if res.status_code == 404:
            raise TosException(res, "Not Found")
        if res.status_code != 206:
            raise TosException(res)
        return Response(res)

    def head_object(self, key, version_id=None):
        assert_validate_key(key)
        query = None
        if version_id:
            query = {'versionId': version_id}
        res = self._req("HEAD", key, query=query, client_method=HeadObjectMethod)
        # print('code:', res.status_code, res.headers)
        if res.status_code == 404:
            raise TosException(res, "Not Found")
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def set_object_meta(self, key, version_id=None, headers=None):
        assert_validate_key(key)
        query = {"metadata": ""}
        if version_id:
            query.update({'versionId': version_id})
        res = self._req("POST", key, headers=headers, query=query, client_method=SetObjectMetaMethod)
        if res.status_code == 404:
            raise TosException(res, "Not Found")
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def put_object_tagging(self, key, tagging_value, version_id=None):
        assert_validate_key(key)
        query = {"tagging": ""}
        if version_id:
            query.update({'versionId': version_id})
        tagging_param = to_put_tagging(tagging_value.encode_kv())
        res = self._req("PUT", key, tagging_param, query=query, client_method=PutObjectTaggingMethod)
        if res.status_code == 404:
            raise TosException(res)
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def get_object_tagging(self, key, version_id=None):
        assert_validate_key(key)
        query = {"tagging": ""}
        if version_id:
            query.update({'versionId': version_id})
        res = self._req("GET", key, query=query, client_method=GetObjectTaggingMethod)
        if res.status_code == 404:
            raise TosException(res)
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def delete_object_tagging(self, key, version_id=None):
        assert_validate_key(key)
        query = {"tagging": ""}
        if version_id:
            query.update({'versionId': version_id})
        res = self._req("DELETE", key, query=query, client_method=DeleteObjectTaggingMethod)
        if res.status_code == 404:
            raise TosException(res)
        if res.status_code != 204:
            raise TosException(res)
        return Response(res)

    def delete_object(self, key, version_id=None):
        assert_validate_key(key)
        query = None
        if version_id:
            query = {'versionId': version_id}
        res = self._req("DELETE", key, query=query, client_method=DeleteObjectMethod)
        if res.status_code in (404, 410):
            raise TosException(res, "Not Found")
        if res.status_code != 204:
            raise TosException(res)
        return Response(res)

    def init_upload(self, key, headers=None):
        assert_validate_key(key)
        res = self._req("POST", key, headers=headers, query={'uploads': ''}, client_method=InitMultipartUploadMethod)
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def upload_part(self, key, upload_id, part_number, body):
        assert_validate_key(key)
        md5hash = None
        crc64W = None
        if self.enable_crc64:
            crc64W = crc64data(self._read(body))
        else:
            md5hash = md5data(self._read(body))

        res = self._req("PUT", key, body=body, query={'partNumber': str(part_number), 'uploadID': upload_id},
                        client_method=UploadPartMethod)
        if res.status_code != 200:
            raise TosException(res)

        if crc64W is not None and self.enable_crc64 \
                and res.headers.get(TOS_HASH_CRC64_ECMA) is not None \
                and res.headers.get(TOS_HASH_CRC64_ECMA) != ""\
                and res.headers.get(TOS_HASH_CRC64_ECMA) != crc64W:
            raise TosException(res, "expect: %s; actual: %s" % (crc64W, res.headers.get(TOS_HASH_CRC64_ECMA)))
        if md5hash is not None \
                and res.headers.get(TOS_MD5_HEADER) is not None \
                and res.headers.get(TOS_MD5_HEADER) != "" \
                and res.headers.get(TOS_MD5_HEADER) != md5hash:
            raise TosException(res, "expect:%s; actual:%s" % (md5hash, res.headers.get(TOS_MD5_HEADER)))
        # etag = res.headers.get(TOS_ETAG_HEADER)
        # if etag:
        #     return part_number + ":" + etag
        return Response(res, part_number=str(part_number))

    def list_parts(self, key, upload_id):
        assert_validate_key(key)
        query = {'uploadID': upload_id}
        res = self._req("GET", key, query=query, client_method=ListPartsMethod)
        if res.status_code == 404:
            raise TosException(res, "Not Found")
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def abort_upload(self, key, upload_id):
        assert_validate_key(key)

        res = self._req("DELETE", key, query={'uploadID': upload_id}, client_method=AbortMultipartUploadMethod)
        if res.status_code != 204:
            raise TosException(res)
        return Response(res)

    def complete_upload(self, key, upload_id, part_list, headers=None):
        assert_validate_key(key)
        body = ",".join(part_list).encode("utf-8")
        res = self._req("POST", key, body=body, headers=headers, query={'uploadID': upload_id},
                        client_method=CompleteMultipartUploadMethod)
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def list_prefix(self, prefix, delimiter, start_after, max_keys):
        """
        :type prefix: str
        :type delimiter:  str
        :type start_after: str
        :type max_keys: int
        :rtype: Response
        """
        res = self._req("GET", "", query={
            "prefix": prefix,
            "delimiter": delimiter,
            "start-after": start_after,
            "max-keys": max_keys,
        }, client_method=ListPrefixMethod)

        if res.status_code != 200:
            raise TosException(res)

        return Response(res)

    def copy_object(self, src_object, dst_object, headers=None):
        """
        在同一个bucket内copy object
        :param src_object: source object
        :type src_object: str
        :param dst_object: destination object
        :type dst_object: str
        :param headers: http header
        :type headers: dict()
        :return: None
        """
        if headers is None:
            headers = {}
        src_path = path_style(self.bucket, src_object)
        headers.update({'X-Tos-Copy-Source': quote(src_path)})
        signed = self.signer.sign_header('POST', quote(src_path), {}, {}, is_copy=True)
        headers.update(signed)
        res = self._req("POST", dst_object, headers=headers, query={"copyobject": ""}, client_method=CopyObjectMethod)
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def copy_object_from(self, dst_object, src_bucket, src_object, src_credentials, headers=None):
        """
        从另外一个bucket copy
        :type dst_object: str
        :type src_bucket: str
        :type src_object: str
        :type src_credentials: Credentials
        :param headers: http header
        :type headers: dict()
        :rtype: Response
        """
        if headers is None:
            headers = {}
        src_path = path_style(src_bucket, src_object)
        headers.update({'X-Tos-Copy-Source': quote(src_path)})
        signer = make_signer(src_credentials, self.sign_version, self.region)
        signed = signer.sign_header('POST', quote(src_path), {}, {}, is_copy=True)
        headers.update(signed)
        res = self._req('POST', dst_object, headers=headers, query={'copyobjectfrom': ''},
                        client_method=CopyObjectMethod)
        if res.status_code != 200:
            raise TosException(res)
        return Response(res)

    def signature_subdomain_link(self, subdomain, key, sig_name, ttl):
        """
        生成一个子域名临时链接
        :param subdomain: str 子域名
        :param obj: str 对象名
        :param sig_name: str 链接名字，取一个名字方便审计，这个名字不必要包含对象名, 字符：0-9 a-z A-z - _
        :param expired_at: int, unix_timestamp，过期时间
        :return: str 链接
        """
        signed = self.signer.sign_query("GET", path_style(self.bucket, key), {}, {}, sign_name=sig_name, ttl=ttl)
        return 'http://' + self.bucket + '.' + subdomain + '/' + quote(key) + '?' + urlencode(signed)

    def list_prefix_versions(self, prefix, delimiter, key_marker, max_keys=100, versionid_marker=""):
        """
        :param prefix: 前缀
        :param delimiter: 对象名称分组的字符
        :param key_marker: 此次列举对象的起点
        :param max_keys: 最大返回数
        :param versionid_marker: 版本号分页标识
        :return:
        """
        resp = self._req("GET", "", query={
            "prefix": prefix,
            "delimiter": delimiter,
            "key-marker": key_marker,
            "max-keys": max_keys,
            "version-id-marker": versionid_marker,
            "versions": ''}, client_method=ListPrefixVersionsMethod)

        if resp.status_code != 200:
            raise TosException(resp)

        return Response(resp)

    def _uri(self, key, **kwargs):
        param = {"timeout": self.timeout}
        param.update(kwargs)
        if self.endpoint and self._is_endpoint_valid():
            return "/%s?%s" % (key, urlencode(param))
        return "/%s/%s?%s" % (self.bucket, key, urlencode(param))

    def _is_endpoint_valid(self):
        if self.force_endpoint:
            return True
        for item in CN_AVAILABLE_ENDPOINTS:
            if self.endpoint == item:
                return True
        for item in I18N_AVAILABLE_ENDPOINTS:
            if self.endpoint == item:
                return True
        return False

    def _req_url(self, bucket, key, query):
        if self.endpoint and self._is_endpoint_valid():
            if key:
                return "/{}?{}".format(key, urlencode(query))
            return "?{}".format(urlencode(query))
        if key:
            return "/{}/{}?{}".format(bucket, key, urlencode(query))
        return "/{}?{}".format(bucket, urlencode(query))

    def _get_host(self):
        addr = self._get_addr()
        host = '{}:{}'.format(*addr)

        if self.endpoint and self._is_endpoint_valid():
            host = addr[0]
        return host

    def _get_url(self, host, key, q):
        url = "http://{}".format(host) + self._req_url(self.bucket, quote(key, safe='/~'), q)
        if self.enable_https:
            url = "https://{}".format(host) + self._req_url(self.bucket, quote(key, safe='/~'), q)

        return url

    def _req(self, method, key, body=None, headers=None, query=None, client_method=None):
        h = CaseInsensitiveDict({HTTP_USER_AGENT_HEADER: self.user_agent})
        if self.remote_psm:
            h['X-Tos-Remote-PSM'] = self.remote_psm

        if headers:
            h.update(headers)
        if not h.get('Accept-Encoding'):
            h['Accept-Encoding'] = ''

        q = {"timeout": '{}s'.format(self.timeout)}
        if query:
            q.update(query)

        host = self._get_host()
        url = self._get_url(host, key, q)

        h['host'] = host
        signed = self.signer.sign_header(method, path_style(self.bucket, key), q, h)
        h.update(signed)
        # print(method, _path_style(self.bucket, key), q, h)
        span = self._before_trace()
        retry_count = self.max_retry_count
        rsp = None
        for i in range(0, retry_count + 1):
            # 采用指数避让策略
            if i != 0:
                sleep_time = SLEEP_BASE_TIME * math.pow(2, i - 1)
                time.sleep(sleep_time)
            can_retry = False
            rsp = self.session.request(method, url, data=body, headers=h,
                                       timeout=self.connect_timeout,stream=self.stream)
            if rsp.status_code >= 500 or rsp.status_code == 429:
                can_retry = _is_func_can_retry(client_method)
            if can_retry:
                host = self._get_host()
                url = self._get_url(host, key, q)
                continue
            else:
                break
        try:
            self._after_trace(span, client_method, host, rsp)
        except Exception as e:
            print("after trace err:", file=sys.stderr)
        return rsp

    def _get_user_agent(self):
        user_agent = 'bytedtos-python-sdk/{0}({1}/{2}/{3};{4})'.format(
            __version__, platform.system(), platform.release(), platform.machine(), platform.python_version())
        return user_agent

    def _get_addr(self):
        if self.addrs:
            return random.choice(self.addrs)

        addr = self._get_env_addr()
        if addr:
            return addr

        addr = self._get_endpoint_addr()
        if addr:
            return addr

        if self.disable_bns:
            addr = self._get_new_addr_without_bns()
            if addr:
                return addr

        return self._get_new_addr_from_bns()

    def _get_one_name(self, name, cluster=None):
        hosts = lookup_name(name, cluster=cluster)
        if not hosts:
            raise TosException("hosts is empty, psm name is %s" % name)
        return self.rd.choice(hosts)

    def _get_new_addr_without_bns(self):
        consul_name = self.service_name
        if self.idc:
            consul_name += ".service." + self.idc

        addr = self._get_one_name(consul_name, cluster=self.cluster)
        port = int(addr["Port"])
        host = addr["Host"]
        if is_ipv6(host):
            host = "[%s]" % host
        return host, port

    def _get_env_addr(self):
        _test_addr = os.environ.get("TEST_TOSAPI_ADDR")
        if _test_addr:
            host_port = _test_addr.split(":")
            return host_port[0], int(host_port[1])

    def _get_endpoint_addr(self):
        if self.endpoint and self._is_endpoint_valid():
            return self.bucket + "." + self.endpoint, 80

    def _before_trace(self):
        return bytedtrace.start_client_span(
            operation_name='resource.tos.' + self.bucket,
            emit_metrics=True,
            emit_log=False,
        )

    def _after_trace(self, span, client_method, host, rsp):
        # 设置 span 属性
        bytedtrace.set_service_type(span, "tos")

        method = client_method or ""
        bytedtrace.set_to_method(span, method)
        bytedtrace.set_to_cluster(span, self.cluster)
        bytedtrace.set_to_addr(span, host)
        if self.idc != "":
            bytedtrace.set_to_dc(span, self.idc)

        if rsp is not None:
            status_code = int(rsp.status_code)
            if status_code >= 500:
                data = json.loads(rsp.content)
                err = data.get('error') or {}
                err_code = err.get('code')
                bytedtrace.set_business_status_code(span, err_code)
                bytedtrace.set_is_error(span, True)
                if not span.is_sampled():
                    span.set_post_trace()
            bytedtrace.set_status_code(span, rsp.status_code)
            span.set_tag("reqID", rsp.headers.get(ReqIdHeader, ""))
            bytedtrace.set_recv_size(span, rsp.headers.get(ContentLengthHeader, 0))

        if span.is_sampled():
            bytedtrace.set_component(span, BytedTraceComponent)
        span.finish()


if __name__ == "__main__":

    _test_client = Client("<bucket>", "<YourAccessKey>", cluster="default", timeout=100, connect_timeout=600)
    # 新账号鉴权模型，仅支持国内BOE环境
    # _test_client = Client("<bucket>", bytedtos.StaticCredentials("<YourAccessKey>", "<YourSecretKey>"), cluster="default", timeout=60, connect_timeout=60)

    value = b'hello world'
    object_name = "test-key"

    # 创建对象标签
    rule = TaggingRule()
    rule.add('key1', 'value1')
    rule.add('-=-', '56.-')

    # 查看编码后的对象标签
    print(rule.to_query_string())

    # 上传对象
    print(_test_client.put_object(object_name, value).headers)

    # 添加或者更新已存在对象的标签信息
    result = _test_client.put_object_tagging(object_name, rule)
    print(result)

    # 获取已存在对象的标签信息
    resp = _test_client.get_object_tagging(object_name)
    print(resp.json)

    # 删除已有对象（Object）的标签（Tagging）信息
    resp = _test_client.delete_object_tagging(object_name)
    print(resp.status_code)

    # 下载对象
    print(_test_client.get_object(object_name).data)
    print(_test_client.head_object(object_name).size, 11)

    # 获取对象元信息
    print(_test_client.head_object(object_name))

    # 设置对象元信息
    _test_client.set_object_meta(object_name, headers={"content-type": "image/png"})
    print(_test_client.head_object(object_name).headers)

    # Range获取对象
    print(_test_client.get_object_range(object_name, 1, 3).data, b"ell")

    # 删除对象
    print(_test_client.delete_object(object_name), True)
    try:
        _test_client.get_object(object_name)
    except TosException as e:
        print(e.code, 404)

    # 分片上传时添加对象标签,设置tagging字符串
    upload_tagging = "key1=value1& key2=value2"

    # 通过HTTP Header的方式设置标签且标签中包含任意字符时，您需要对标签的Key和Value进行URL编码。
    key3 = "k3+-="
    value3 = "+-=._:/"
    upload_tagging += "&" + quote(key3) + "=" + quote(value3)

    # 调用init_upload接口初始化分片时指定headers，将会给上传的文件添加标签。
    upload_headers = dict()
    upload_headers[TosObjectTaggingHeader] = upload_tagging

    upload_object_key = "upload-test-key"
    # payload_id = _test_client.init_upload(upload_object_key, upload_headers).upload_id
    payload_id = _test_client.init_upload(upload_object_key).upload_id
    print(payload_id)
    n = 3
    large_value = value * 1000 * 1000
    tmp_part_lists = []
    for i in range(n):
        upload_resp = _test_client.upload_part(upload_object_key, payload_id, i+1, large_value)
        print(upload_resp.part_number)
        tmp_part_lists.append(upload_resp.part_number)

    list_part_resp = _test_client.list_parts(upload_object_key, payload_id)
    _test_client.complete_upload(upload_object_key, payload_id, tmp_part_lists)
    print(_test_client.get_object(upload_object_key).data == large_value * n)

    # 本地文件上传
    _test_client.put_object("test-key-file", open("../tests/data", "r"))
    print(_test_client.get_object("test-key-file").data, b'hello world')

    # 流式上传
    _test_client.stream = True
    resp = _test_client.get_object("test-key-file").raw
    print(resp.read(3), b'hel')
    print(resp.read(2), b'lo')

    # 终止分片上传
    payload_id = _test_client.init_upload(upload_object_key).upload_id
    resp = _test_client.abort_upload(upload_object_key, payload_id)
    print(resp)

    # 列举bucket中的对象
    resp = _test_client.list_prefix("abc", "/", "abcd", 10)
    print(resp.data)

    # 桶内拷贝时添加对象标签
    copy_value = b'hello world'
    src_object_name = "test-key-1"
    dst_object_name = "copy-test-key-1"
    _test_client.put_object(src_object_name, copy_value)
    _test_client.put_object(dst_object_name, copy_value)

    # 设置tagging字符串。
    tagging = "key1=value1& key2=value2"

    # 通过HTTP Header的方式设置标签且标签中包含任意字符时，您需要对标签的Key和Value进行URL编码。
    key3 = "k3+-="
    value3 = "+-=._:/"
    tagging += "&" + quote(key3) + "=" + quote(value3)

    copy_headers = dict()
    copy_headers[TosObjectTaggingHeader] = tagging

    # 指定设置目标Object对象标签的方式。此处默认设置为Copy，表示复制源Object的对象标签到目标Object。
    # copy_headers[CopyTaggingDirectiveHeader] = "Copy"
    # 此处设置为Replace，表示忽略源Object的对象标签，采用请求头中对象标签到目标Object。
    copy_headers[CopyTaggingDirectiveHeader] = "Replace"
    print(_test_client.get_object_tagging(src_object_name).json)
    resp = _test_client.copy_object(src_object_name, dst_object_name, copy_headers)

    # 桶内拷贝，不配置自定义header
    # resp = _test_client.copy_object(src_object_name, dst_object_name)
    print(_test_client.get_object(dst_object_name).data == value)
    print(_test_client.get_object_tagging(dst_object_name).json)

    # 多版本操作
    _version_test_client = Client("<bucket>", "<YourAccessKey>", cluster="default", timeout=60, connect_timeout=60)
    value1 = b'hello world v1'
    value2 = b'hello world v2'
    value3 = b'hello world v3'
    multi_version_object_name = "mv-test-key"

    resp = _version_test_client.put_object(multi_version_object_name, value1)
    version1 = resp.headers.get('X-Tos-Version-Id')
    print(version1)

    # 创建对象标签
    mv_rule = TaggingRule()
    mv_rule.add('key1', 'value1')
    mv_rule.add('-=-', '56.-')

    # 多版本桶：添加或者更新已存在对象的标签信息
    result = _version_test_client.put_object_tagging(multi_version_object_name, mv_rule, version_id=version1)
    print(result)

    # 多版本桶：获取已存在对象的标签信息, 默认只能获取Object当前版本的标签信息
    resp_default = _version_test_client.get_object_tagging(multi_version_object_name)
    print(resp_default.json)

    # 多版本桶：按照version id获取已存在对象的标签信息
    resp_with_id = _version_test_client.get_object_tagging(multi_version_object_name, version_id=version1)
    print(resp_default.json == resp_with_id.json)

    # 删除已有对象（Object）的标签（Tagging）信息
    resp = _version_test_client.delete_object_tagging(multi_version_object_name)
    print(resp.status_code)

    resp = _version_test_client.put_object(multi_version_object_name, value2)
    version2 = resp.headers.get('X-Tos-Version-Id')
    print(version2)

    resp = _version_test_client.put_object(multi_version_object_name, value3)
    version3 = resp.headers.get('X-Tos-Version-Id')
    print(version3)

    """
    列出前缀为test-key的所有版本
    """
    is_truncated = True  # 用于判断列举是否结束
    multi_version_prefix = "test-key"
    key_marker = ""  # 首次列举设置为空
    versionid_marker = ""  # 首次列举设置为空
    while is_truncated:
        resp = _version_test_client.list_prefix_versions(multi_version_prefix, "", key_marker, 2, versionid_marker)
        data = json.loads(resp.data)

        for obj in data['payload']['objects']:
            print(obj['key'])
            print(obj['versionId'])
            print(obj['deleteMarker'])  # 是否为删除版本
            print(obj['isLatest'])  # 是否为最新版本

        is_truncated = data['payload']['isTruncated']
        key_marker = data['payload']['startAfter']
        versionid_marker = data['payload']['startAfterVersionId']

    resp = _version_test_client.head_object(multi_version_object_name, version_id=version1)
    print("head: ", resp.data)

    resp = _version_test_client.get_object(multi_version_object_name, version_id=version1)
    print("get:", resp.data, value1)

    print(_version_test_client.delete_object(multi_version_object_name, version_id=version1).headers)
    # 多版本桶：按照version id获取已存在对象的标签信息
    # 如果Object的对应版本为删除标记（Delete Marker），则将返回404 Not Found
    try:
        resp_with_id = _version_test_client.get_object_tagging(multi_version_object_name, version_id=version1)
    except TosException as e:
        print(e.code, 404)
    print(_version_test_client.delete_object(multi_version_object_name, version_id=version2).headers)
    print(_version_test_client.delete_object(multi_version_object_name, version_id=version3).headers)
